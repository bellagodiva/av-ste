# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Modified for Mimi semantic token prediction:
# AV-HuBERT hidden states -> linear / MLP head -> Mimi semantic token logits

import sys, logging
import contextlib
from argparse import Namespace
from typing import Any

import torch
import torch.nn as nn
from dataclasses import dataclass, field

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING
from avhubert.hubert_asr import AVHubertAsrConfig, HubertEncoderWrapper

DBG = True if len(sys.argv) == 1 else False

if DBG:
    from hubert import AVHubertModel
else:
    from ..hubert import AVHubertModel

logger = logging.getLogger(__name__)


@dataclass
class AVHubertMimiConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to AV-HuBERT model"}
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside AV-HuBERT model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability for attention weights inside AV-HuBERT model"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN inside AV-HuBERT model"},
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={"help": "probability of replacing a token with mask (normalized by length)"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0.0,
        metadata={"help": "secondary mask argument used for more complex distributions"},
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0.0,
        metadata={"help": "secondary mask argument used for more complex distributions"},
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )

    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "don't finetune AV-HuBERT for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in AV-HuBERT to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in AV-HuBERT"},
    )

    # Mimi head
    mimi_vocab_size: int = field(
        default=2048,
        metadata={"help": "number of Mimi semantic token IDs"},
    )
    head_hidden_dim: int = field(
        default=0,
        metadata={"help": "0 = linear head, >0 = 2-layer MLP hidden dim"},
    )

    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    label_dir: str = II("task.label_dir")

    # holds loaded AV-HuBERT args
    w2v_args: Any = None


class MimiHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                Linear(in_dim, out_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                Linear(hidden_dim, out_dim),
            )

    def forward(self, x):
        return self.net(x)


class HubertEncoder(FairseqEncoder):
    """
    Reuses pretrained AV-HuBERT encoder and adds a frame-wise Mimi prediction head.
    """

    def __init__(self, cfg: AVHubertMimiConfig):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
        w2v_args = state.get("cfg", None)
        if w2v_args is None:
            w2v_args = convert_namespace_to_omegaconf(state["args"])
        elif isinstance(w2v_args, Namespace):
            w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here."
        )

        # Use current dataset path
        w2v_args.task.data = cfg.data

        # Build pretraining task exactly like the reference code
        task_pretrain = tasks.setup_task(w2v_args.task)
        if "task_state" in state and state["task_state"] is not None:
            task_pretrain.load_state_dict(state["task_state"])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        # Same pattern as the reference code
        avhubert = HubertEncoderWrapper(encoder_)

        if not cfg.no_pretrained_weights:
            sd = state["model"].copy()

            # incompatible / unnecessary for Mimi finetuning
            sd.pop("mask_emb", None)
            sd.pop("label_embs_concat", None)

            # load only into underlying AV-HuBERT model
            missing, unexpected = avhubert.w2v_model.load_state_dict(sd, strict=False)
            print(f"[HubertEncoder] missing keys: {len(missing)}")
            print(f"[HubertEncoder] unexpected keys: {len(unexpected)}")

        avhubert.w2v_model.remove_pretraining_modules()

        super().__init__(task_pretrain.source_dictionary)

        # embedding dim from the wrapped AV-HuBERT model
        d = avhubert.w2v_model.encoder.embedding_dim

        self.w2v_model = avhubert.w2v_model
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        # 25 Hz -> 12.5 Hz
        self.temporal_downsample = nn.Conv1d(
            in_channels=d,
            out_channels=d,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.mimi_head = MimiHead(
            in_dim=d,
            out_dim=cfg.mimi_vocab_size,
            hidden_dim=cfg.head_hidden_dim,
            dropout=cfg.final_dropout,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def downsample_padding_mask(self, padding_mask):
        """
        padding_mask: [B, T], True = padded
        returns: [B, T//2]
        """
        if padding_mask is None:
            return None

        if padding_mask.size(1) % 2 == 1:
            padding_mask = padding_mask[:, :-1]

        return padding_mask[:, 0::2] & padding_mask[:, 1::2]

    def upsample_by_repeat(self, x, target_len=None):
        """
        x: [B, T, ...] -> [B, 2T, ...] by repeating each frame twice
        Example: 33 45 67 21 -> 33 33 45 45 67 67 21 21
        """
        x = x.repeat_interleave(2, dim=1)

        if target_len is not None:
            cur_len = x.size(1)
            if cur_len > target_len:
                x = x[:, :target_len, ...]
            elif cur_len < target_len:
                pad_len = target_len - cur_len
                last = x[:, -1:, ...].expand(x.size(0), pad_len, *x.shape[2:])
                x = torch.cat([x, last], dim=1)
        return x


    def upsample_padding_mask_by_repeat(self, padding_mask, target_len=None):
        """
        padding_mask: [B, T] -> [B, 2T]
        """
        if padding_mask is None:
            return None

        padding_mask = padding_mask.repeat_interleave(2, dim=1)

        if target_len is not None:
            cur_len = padding_mask.size(1)
            if cur_len > target_len:
                padding_mask = padding_mask[:, :target_len]
            elif cur_len < target_len:
                pad_len = target_len - cur_len
                last = padding_mask[:, -1:].expand(padding_mask.size(0), pad_len)
                padding_mask = torch.cat([padding_mask, last], dim=1)

        return padding_mask

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)

        # Save original 25 Hz length before any even-length trimming
        orig_len = x.size(1)
        orig_padding_mask = padding_mask
        # x: B x T x C at ~25 Hz, make length even before stride-2 downsampling
        if x.size(1) % 2 == 1:
            x = x[:, :-1, :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-1]
        trimmed_len = x.size(1)
        # downsample features: B x T x C -> B x T/2 x C
        x = x.transpose(1, 2)                   # B x C x T
        x = self.temporal_downsample(x)         # B x C x T/2
        x = x.transpose(1, 2)                   # B x T/2 x C

        padding_mask = self.downsample_padding_mask(padding_mask)

        x = self.final_dropout(x)
        logits_12p5 = self.mimi_head(x)              # B x T/2 x V

        # Upsample back to 25 Hz so logits align with labels during both
        # training and validation CE loss computation.
        logits = self.upsample_by_repeat(logits_12p5, target_len=orig_len)
        out_padding_mask = self.upsample_padding_mask_by_repeat(
            padding_mask, target_len=orig_len
        )

        if tbc:
            logits = logits.transpose(0, 1)  # T/2 x B x V

        return {
            "encoder_out": logits,                 # T x B x V or B x T x V
            "encoder_padding_mask": out_padding_mask, # B x T
            "padding_mask": out_padding_mask,         # B x T
            "features": x,                        # B x T x C (before classifier)
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            # encoder_out is T x B x V
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out["encoder_padding_mask"].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(0, new_order)
        if encoder_out.get("features", None) is not None:
            encoder_out["features"] = encoder_out["features"].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


@register_model("av_hubert_mimi", dataclass=AVHubertMimiConfig)
class AVHubertMimi(BaseFairseqModel):
    def __init__(self, cfg: AVHubertMimiConfig, encoder: HubertEncoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder

    @classmethod
    def build_model(cls, cfg: AVHubertMimiConfig, task: FairseqTask):
        encoder = HubertEncoder(cfg)
        return cls(cfg, encoder)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.encoder.set_num_updates(num_updates)

    def get_normalized_probs(self, net_output, log_probs):
        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T  # B x T -> T x B
            logits[padding] = float("-inf")
        return logits

    def forward(self, **kwargs):
        return self.encoder(**kwargs)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

"""
fairseq-hydra-train    
--config-dir /mnt/hard1/bella/EMNLP26/source/av_hubert/avhubert/conf/finetune     
--config-name large_lrs3_433h_mimi.yaml     
task.data=/mnt/hard1/bella/EMNLP26/dataset/LRS3/433h 
task.label_dir=/mnt/hard1/bella/EMNLP26/dataset/LRS3/433h 
hydra.run.dir=/mnt/hard1/bella/EMNLP26/experiment/av_10kfreeze     
common.user_dir=/mnt/hard1/bella/EMNLP26/source/av_hubert/avhubert

"""