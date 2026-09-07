import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import metrics, utils


@register_criterion("mimi_frame_ce_last")
class MimiFrameCriterion(FairseqCriterion):
    """
    Frame-wise cross entropy for Mimi semantic token prediction.

    The model predicts at 12.5 Hz internally but upsamples logits to 25 Hz
    (repeat_interleave x2) during training so that fairseq's integer label_rate
    constraint is satisfied.  Each unique logit therefore appears twice against
    an identical target, which:
      - makes the summed CE loss 2x the true 12.5 Hz loss (but sample_size is
        also 2x, so the logged loss/token scalar is correct and comparable)
      - inflates a naive frame-count accuracy by ~2x (both copies of a logit
        are evaluated against the same token, so a correct prediction counts
        twice)

    To get an honest accuracy we evaluate only every other frame (the "anchor"
    frame, index 0::2), which corresponds to the original 12.5 Hz rate.  We
    also log the raw 25 Hz accuracy for reference; the two should be nearly
    identical -- a significant divergence signals a bug in the upsampling logic.
    """

    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = -100

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _pad_targets(self, target_list, device):
        """
        target_list : list of per-sample 1-D (or trivially-squeezable 2-D) tensors
        returns     : LongTensor [B, T_max], padded with self.padding_idx
        """
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

    @staticmethod
    def _to_btv(logits, batch_size):
        """
        Ensure logits are [B, T, V].
        Accepts both [T, B, V] (tbc=True, the default) and [B, T, V].
        Raises if neither dimension matches batch_size.
        """
        if logits.dim() != 3:
            raise ValueError(f"Expected 3-D logits, got shape {tuple(logits.shape)}")
        if logits.size(1) == batch_size:          # [T, B, V]
            return logits.transpose(0, 1)
        elif logits.size(0) == batch_size:         # [B, T, V]
            return logits
        else:
            raise ValueError(
                f"Cannot infer logits layout. "
                f"logits={tuple(logits.shape)}, batch_size={batch_size}"
            )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        raw_logits = net_output["encoder_out"]   # [T,B,V] or [B,T,V]

        # ---- parse targets ----------------------------------------
        raw = sample["target_list"]
        if not isinstance(raw, list):
            raise TypeError(f"target_list must be a list, got {type(raw)}")

        if len(raw) == 1 and isinstance(raw[0], list):
            # one label stream -> list of per-sample targets
            target = self._pad_targets(raw[0], raw_logits.device)
        elif len(raw) == 1 and torch.is_tensor(raw[0]):
            # one label stream -> already batched [B, T]
            target = raw[0].to(device=raw_logits.device, dtype=torch.long)
            if target.dim() == 1:
                target = target.unsqueeze(0)
            elif target.dim() != 2:
                raise ValueError(
                    f"Expected batched target [B,T], got {tuple(target.shape)}"
                )
        else:
            # list of per-sample tensors / lists
            assert all(
                (torch.is_tensor(r) and r.dim() <= 2) or isinstance(r, (list, torch.Tensor))
                for r in raw
            ), f"Unexpected target_list format: {[type(r) for r in raw]}"
            target = self._pad_targets(raw, raw_logits.device)

        B = target.size(0)

        # ---- normalise logits to [B, T, V] --------------------------
        logits = self._to_btv(raw_logits, B)
        _, T, V = logits.shape
        Tt = target.size(1)

        # reconcile length (25 Hz logits vs 25 Hz duplicated labels)
        if Tt > T:
            target = target[:, :T]
        elif Tt < T:
            pad = torch.full(
                (B, T - Tt), self.padding_idx,
                dtype=target.dtype, device=target.device,
            )
            target = torch.cat([target, pad], dim=1)

        # ---- loss (summed CE over 25 Hz frames) ---------------------
        logits_flat = logits.reshape(B * T, V)          # [B*T, V]
        target_flat = target.reshape(B * T)             # [B*T]

        loss = F.cross_entropy(
            logits_flat,
            target_flat,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

        non_pad_mask = target_flat != self.padding_idx
        # sample_size = number of valid 25 Hz frames (2x the true token count)
        # loss / sample_size therefore gives the correct per-token CE scalar
        sample_size = non_pad_mask.sum().item()

        # ---- metrics ------------------------------------------------
        logging_output = {
            "loss":        loss.detach().item() if torch.is_tensor(loss) else loss,
            "ntokens":     sample_size,
            "nsentences":  B,
            "sample_size": sample_size,
        }

        with torch.no_grad():
            pred_flat = logits_flat.argmax(dim=-1)      # [B*T]

            # --- 25 Hz accuracy (all frames, inflated by duplication) ---
            correct_25 = ((pred_flat == target_flat) & non_pad_mask).sum().item()
            total_25   = non_pad_mask.sum().item()

            # --- 12.5 Hz accuracy (anchor frames only, honest metric) ---
            # Reshape to [B, T] and stride by 2 to recover original token rate.
            # Because the model uses repeat_interleave(2), frame pairs
            # (2k, 2k+1) always share the same logit and the same target.
            # Taking every even frame (0::2) gives one evaluation per unique
            # prediction -- this is the number that should be compared against
            # other systems that compute accuracy at the true 12.5 Hz rate.
            pred_2d   = pred_flat.reshape(B, T)
            target_2d = target.reshape(B, T)

            pred_12   = pred_2d[:, 0::2].reshape(-1)    # [B * T/2]
            tgt_12    = target_2d[:, 0::2].reshape(-1)  # [B * T/2]

            non_pad_12 = tgt_12 != self.padding_idx
            correct_12 = ((pred_12 == tgt_12) & non_pad_12).sum().item()
            total_12   = non_pad_12.sum().item()

            logging_output["correct"]    = correct_12   # honest 12.5 Hz figure
            logging_output["total"]      = total_12
            logging_output["correct_25"] = correct_25   # reference / sanity check
            logging_output["total_25"]   = total_25
            # Sanity: if correct_12/total_12 diverges greatly from
            # correct_25/total_25, the repeat_interleave upsampling has a bug.

        return loss, sample_size, logging_output

    # ------------------------------------------------------------------
    # aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def reduce_metrics(logging_outputs):
        loss_sum    = sum(log.get("loss", 0)        for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        correct_12  = sum(log.get("correct", 0)     for log in logging_outputs)
        total_12    = sum(log.get("total", 0)        for log in logging_outputs)
        correct_25  = sum(log.get("correct_25", 0)  for log in logging_outputs)
        total_25    = sum(log.get("total_25", 0)    for log in logging_outputs)

        if torch.is_tensor(loss_sum):
            loss_sum = utils.item(loss_sum)

        # loss/token -- the 2x from duplication cancels with the 2x sample_size
        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

        # honest 12.5 Hz accuracy -- use this as your primary training metric
        if total_12 > 0:
            metrics.log_scalar(
                "accuracy",
                correct_12 / total_12,
                total_12,
                round=6,
            )

        # 25 Hz accuracy -- should be ~equal to accuracy above
        # a large gap means repeat_interleave is producing inconsistent pairs
        if total_25 > 0:
            metrics.log_scalar(
                "accuracy_25hz",
                correct_25 / total_25,
                total_25,
                round=6,
            )