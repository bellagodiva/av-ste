# Frame-wise cross-entropy criterion for all 8 Mimi RVQ codebooks.
#
# Expected model output:
#   net_output["encoder_out"]  shape [T, B, num_rvq, V]  (tbc=True)
#                           or shape [B, T, num_rvq, V]
#
# Expected sample["target_list"]:
#   A list of num_rvq elements, each being a [B, T] LongTensor or a list of
#   1-D per-sample LongTensors — one element per RVQ codebook.
#   (Corresponds to labels: ["mimi0", "mimi1", ..., "mimi7"] in the YAML.)
#
# Loss = mean of num_rvq per-codebook cross-entropies, each summed over tokens.
# Accuracy is logged for codebook 0 (semantic) and averaged across all codebooks.

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq import metrics, utils


@dataclass
class MimiRVQCEConfig(FairseqDataclass):
    num_rvq: int = field(
        default=8,
        metadata={"help": "number of RVQ codebooks (must match model.num_rvq)"},
    )
    # Optional per-codebook loss weights. If empty, all codebooks are weighted equally.
    # Pass as a comma-separated string, e.g. "2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0"
    codebook_weights: str = field(
        default="",
        metadata={
            "help": (
                "Comma-separated per-codebook loss weights (length must equal num_rvq). "
                "Empty string = uniform weighting."
            )
        },
    )


@register_criterion("mimi_rvq_ce", dataclass=MimiRVQCEConfig)
class MimiRVQCECriterion(FairseqCriterion):
    """
    Sums num_rvq independent frame-wise cross-entropy losses.
    """

    def __init__(self, cfg: MimiRVQCEConfig, task):
        super().__init__(task)
        self.padding_idx = -100
        self.num_rvq = cfg.num_rvq

        if cfg.codebook_weights.strip():
            weights = [float(w) for w in cfg.codebook_weights.split(",")]
            assert len(weights) == self.num_rvq, (
                f"codebook_weights has {len(weights)} entries but num_rvq={self.num_rvq}"
            )
            self.register_buffer(
                "cb_weights",
                torch.tensor(weights, dtype=torch.float32),
            )
        else:
            self.cb_weights = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pad_targets(self, target_list, device):
        """List of 1-D per-sample tensors → [B, T] padded with padding_idx."""
        proc = []
        for t in target_list:
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.long, device=device)
            else:
                t = t.to(device=device, dtype=torch.long)
            if t.dim() == 2:
                if t.size(0) == 1:
                    t = t.squeeze(0)
                elif t.size(1) == 1:
                    t = t.squeeze(1)
                else:
                    raise ValueError(f"Unexpected target shape: {tuple(t.shape)}")
            elif t.dim() != 1:
                raise ValueError(f"Expected 1-D target per sample, got {tuple(t.shape)}")
            proc.append(t)
        T = max(x.numel() for x in proc)
        out = torch.full((len(proc), T), self.padding_idx, dtype=torch.long, device=device)
        for i, t in enumerate(proc):
            out[i, : t.numel()] = t
        return out

    def _parse_one_codebook_target(self, raw, device):
        """
        raw: either
          - a [B, T] LongTensor
          - a list of 1-D per-sample tensors
        Returns [B, T] LongTensor.
        """
        if torch.is_tensor(raw):
            t = raw.to(device=device, dtype=torch.long)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return t
        # list of per-sample tensors
        return self._pad_targets(raw, device)

    @staticmethod
    def _align_len(target, T, padding_idx, device):
        """Trim or pad target [B, Tt] to length T."""
        Tt = target.size(1)
        if Tt > T:
            return target[:, :T]
        if Tt < T:
            pad = torch.full(
                (target.size(0), T - Tt), padding_idx,
                dtype=target.dtype, device=device,
            )
            return torch.cat([target, pad], dim=1)
        return target

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        logits_raw = net_output["encoder_out"]   # [T, B, num_rvq, V] or [B, T, num_rvq, V]

        if logits_raw.dim() != 4:
            raise ValueError(
                f"Expected 4-D encoder_out [T,B,R,V] or [B,T,R,V], got {tuple(logits_raw.shape)}"
            )

        raw_targets = sample["target_list"]
        if not isinstance(raw_targets, list) or len(raw_targets) != self.num_rvq:
            raise ValueError(
                f"target_list must be a list of {self.num_rvq} elements (one per RVQ codebook), "
                f"got {type(raw_targets)} with len={len(raw_targets) if isinstance(raw_targets, list) else '?'}"
            )

        # Parse codebook-0 target to determine batch size and infer logits layout
        target0 = self._parse_one_codebook_target(raw_targets[0], logits_raw.device)
        B = target0.size(0)

        # Infer [B, T, num_rvq, V] layout
        if logits_raw.size(1) == B:
            # [T, B, num_rvq, V] → [B, T, num_rvq, V]
            logits = logits_raw.transpose(0, 1)
        elif logits_raw.size(0) == B:
            logits = logits_raw
        else:
            raise ValueError(
                f"Cannot determine logits layout: shape={tuple(logits_raw.shape)}, B={B}"
            )

        _, T, num_rvq, V = logits.shape
        assert num_rvq == self.num_rvq, f"logits num_rvq={num_rvq} != cfg.num_rvq={self.num_rvq}"

        weight = self.cb_weights if self.cb_weights is not None else None

        total_loss   = logits.new_zeros(1).squeeze()
        sample_size  = 0
        correct_list = []
        total_list   = []

        for k in range(self.num_rvq):
            if k == 0:
                target_k = target0
            else:
                target_k = self._parse_one_codebook_target(raw_targets[k], logits.device)

            target_k = self._align_len(target_k, T, self.padding_idx, logits.device)

            logits_k = logits[:, :, k, :].contiguous()  # [B, T, V]
            logits_flat = logits_k.reshape(B * T, V)
            target_flat = target_k.reshape(B * T)

            loss_k = F.cross_entropy(
                logits_flat,
                target_flat,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )

            w = weight[k].item() if weight is not None else 1.0
            total_loss = total_loss + w * loss_k

            non_pad = target_flat != self.padding_idx
            n_valid = int(non_pad.sum().item())
            if k == 0:
                sample_size = n_valid

            with torch.no_grad():
                pred = logits_flat.argmax(dim=-1)
                correct_list.append(int(((pred == target_flat) & non_pad).sum().item()))
                total_list.append(n_valid)

        acc_k0  = correct_list[0] / max(total_list[0], 1)
        acc_all = sum(correct_list) / max(sum(total_list), 1)

        logging_output = {
            "loss":        total_loss.detach().item(),
            "ntokens":     sample_size,
            "nsentences":  B,
            "sample_size": sample_size,
            "correct":     correct_list[0],
            "total":       total_list[0],
            "correct_all": sum(correct_list),
            "total_all":   sum(total_list),
        }

        return total_loss, sample_size, logging_output

    # ------------------------------------------------------------------
    # Metric aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def reduce_metrics(logging_outputs):
        loss_sum       = sum(log.get("loss", 0)        for log in logging_outputs)
        sample_size    = sum(log.get("sample_size", 0) for log in logging_outputs)
        correct_k0     = sum(log.get("correct", 0)     for log in logging_outputs)
        total_k0       = sum(log.get("total", 0)       for log in logging_outputs)
        correct_all    = sum(log.get("correct_all", 0) for log in logging_outputs)
        total_all      = sum(log.get("total_all", 0)   for log in logging_outputs)

        if torch.is_tensor(loss_sum):
            loss_sum = utils.item(loss_sum)

        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
        if total_k0 > 0:
            metrics.log_scalar("accuracy",     correct_k0  / total_k0,  total_k0,  round=6)
        if total_all > 0:
            metrics.log_scalar("accuracy_all", correct_all / total_all, total_all, round=6)
