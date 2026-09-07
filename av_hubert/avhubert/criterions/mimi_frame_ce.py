import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import metrics, utils


@register_criterion("mimi_frame_ce")
class MimiFrameCriterion(FairseqCriterion):
    """
    Frame-wise cross entropy for Mimi semantic tokens
    """

    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = -100
    
    def _pad_targets(self, target_list, device):
        """
        target_list: list of per-sample targets
        returns: LongTensor [B, T]
        """
        proc = []
        for t in target_list:
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.long, device=device)
            else:
                t = t.to(device=device, dtype=torch.long)

            # squeeze trivial dims
            if t.dim() == 2:
                if t.size(0) == 1:
                    t = t.squeeze(0)
                elif t.size(1) == 1:
                    t = t.squeeze(1)
                else:
                    raise ValueError(f"Unexpected target shape: {tuple(t.shape)}")
            elif t.dim() != 1:
                raise ValueError(f"Expected 1D target per sample, got {tuple(t.shape)}")

            proc.append(t)

        B = len(proc)
        T = max(x.numel() for x in proc)
        out = torch.full((B, T), self.padding_idx, dtype=torch.long, device=device)
        for i, t in enumerate(proc):
            out[i, : t.numel()] = t
        return out

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        logits = net_output["encoder_out"]   # [T,B,V] or [B,T,V]

        if logits.dim() != 3:
            raise ValueError(f"Expected 3D logits, got {tuple(logits.shape)}")

        raw = sample["target_list"]

        if not isinstance(raw, list):
            raise TypeError(f"target_list should be list, got {type(raw)}")

        # ---- parse target_list robustly ----
        if len(raw) == 1 and isinstance(raw[0], list):
            # one stream -> list of per-sample targets
            target = self._pad_targets(raw[0], logits.device)

        elif len(raw) == 1 and torch.is_tensor(raw[0]):
            # one stream -> already batched tensor [B,T]
            target = raw[0].to(device=logits.device, dtype=torch.long)
            if target.dim() == 1:
                target = target.unsqueeze(0)
            elif target.dim() != 2:
                raise ValueError(f"Expected batched target [B,T], got {tuple(target.shape)}")

        else:
            # assume list of per-sample targets
            target = self._pad_targets(raw, logits.device)

        # ---- convert logits to [B,T,V] if needed ----
        # If encoder_out is [T,B,V], middle dim should match batch
        if logits.size(1) == target.size(0):
            logits = logits.transpose(0, 1)
        elif logits.size(0) == target.size(0):
            pass
        else:
            raise ValueError(
                f"Cannot infer logits layout. logits={tuple(logits.shape)}, target={tuple(target.shape)}"
            )

        B, T, V = logits.shape
        Bt, Tt = target.shape

        if B != Bt:
            raise ValueError(
                f"Batch mismatch: logits {tuple(logits.shape)}, target {tuple(target.shape)}"
            )

        # match target length to logits length
        if Tt > T:
            target = target[:, :T]
        elif Tt < T:
            pad = torch.full(
                (B, T - Tt),
                self.padding_idx,
                dtype=target.dtype,
                device=target.device,
            )
            target = torch.cat([target, pad], dim=1)
        logits_flat = logits.reshape(B * T, V)
        target_flat = target.reshape(B * T)
        loss = F.cross_entropy(
            logits_flat,
            target_flat,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

        non_pad_mask = target_flat != self.padding_idx
        sample_size = non_pad_mask.sum().item()

        logging_output = {
            "loss": loss.detach().item() if torch.is_tensor(loss) else loss,
            "ntokens": sample_size,
            "nsentences": B,
            "sample_size": sample_size,
        }

        with torch.no_grad():
            pred = logits_flat.argmax(dim=-1)
            correct = (pred == target_flat) & non_pad_mask
            logging_output["correct"] = correct.sum().item()
            logging_output["total"] = non_pad_mask.sum().item()
            #print("loss raw:", loss.item(), "sample_size:", sample_size, "loss/sample:", loss.item() / max(sample_size, 1))

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        correct = sum(log.get("correct", 0) for log in logging_outputs)
        total = sum(log.get("total", 0) for log in logging_outputs)

        loss_sum = utils.item(loss_sum) if torch.is_tensor(loss_sum) else loss_sum

        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

        if total > 0:
            metrics.log_scalar("accuracy", correct / total, total, round=6)