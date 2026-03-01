# Copyright (c) 2024, Tencent Inc. All rights reserved.
from typing import Optional, Tuple

import deepspeed
import math
import torch
import torch.nn as nn

from tgnn.config import configurable
from tgnn.model.build import MODEL_REGISTRY
from tgnn.model.layer import RMSNorm, DropPath, Embedding
from tgnn.model.module.attention import MSARowAttention, MSAColumnAttention
from tgnn.model.module.mlp import SwiGLU
from tgnn.tokenizer import build_tokenizer
from tgnn.utils import warn_rank_0
from tgnn.utils.tensor import masked_mean


class USMBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 num_row_kv_heads=None,
                 num_col_kv_heads=None,
                 ffn_dim=None,
                 ffn_dim_multiplier: Optional[float] = None,
                 sliding_window_size: Optional[int] = None,
                 bias: bool = False,
                 droppath: float = 0.0,
                 is_causal: bool = False,
                 dual_path: bool = False,
                 with_column_attention: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.row_attn = MSARowAttention(dim,
                                        num_heads,
                                        is_causal=is_causal,
                                        num_kv_heads=num_row_kv_heads,
                                        sliding_window_size=sliding_window_size,
                                        bias=bias)
        self.with_column_attention = with_column_attention
        if self.with_column_attention:
            self.norm_1 = RMSNorm(dim, eps=eps)
            self.col_attn = MSAColumnAttention(dim,
                                               num_heads,
                                               is_causal=is_causal,
                                               num_kv_heads=num_col_kv_heads,
                                               bias=bias)
        self.mlp = SwiGLU(dim,
                          ffn_dim,
                          ffn_dim_multiplier=ffn_dim_multiplier,
                          bias=bias)
        self.norm_2 = RMSNorm(dim, eps=eps)
        self.norm_3 = RMSNorm(dim, eps=eps)
        self.drop_path = DropPath(droppath)
        self.dual_path = dual_path

    def forward(self,
                m: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attn_weight=False,
                inplace: bool = False,
                chunk_size: int = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            m: [bs, row, col, dim]
            mask: [bs, row, col], attention mask
        """
        if self.with_column_attention:
            hidden, _ = self.col_attn(self.norm_1(m, mask=mask, inplace=inplace),
                                      mask=mask,
                                      inplace=inplace,
                                      chunk_size=chunk_size)
            m = m + self.drop_path(hidden)

        sm = masked_mean(mask[..., None] if mask is not None else None,
                         m.float(),
                         dim=1,
                         keepdim=True).type_as(m)  # [bs, 1, col, dim]
        col_mask = (mask.sum(dim=1, keepdim=True) > 0) if mask is not None else None  # [bs, 1, col]
        hidden, attn_weight = self.row_attn(self.norm_2(sm, mask=col_mask, inplace=inplace),
                                            mask=col_mask,
                                            inplace=inplace,
                                            return_attn_weight=return_attn_weight)
        m = m + hidden
        if self.dual_path:
            skip_mask = col_mask if mask is not None else None
            skip = sm

        else:
            skip = m
            skip_mask = mask if mask is not None else None

        m = m + self.mlp(self.norm_3(skip), mask=skip_mask, inplace=inplace)
        return m, attn_weight


@MODEL_REGISTRY.register()
class USM(nn.Module):
    CONFIG = {
        "msa-30m": dict(num_layers=12, num_heads=12, embedding_dim=384, bias=False),
        "msa-100m": dict(num_layers=12, num_heads=12, embedding_dim=768, bias=False),
        "msa-650m": dict(num_layers=26, num_heads=32, embedding_dim=1280, num_col_kv_heads=8, bias=False),
        "msa-1b": dict(num_layers=29, num_heads=32, embedding_dim=1536, num_col_kv_heads=8, bias=False),
        "msa-3b": dict(num_layers=37, num_heads=32, embedding_dim=2304, num_col_kv_heads=16, bias=False)
    }

    @classmethod
    def from_config(cls, cfg):
        model_type = cfg.model.type
        if model_type:
            mcfg = cls.CONFIG[model_type]
            num_layers, num_heads, embedding_dim = mcfg['num_layers'], mcfg['num_heads'], mcfg['embedding_dim']
            num_row_kv_heads = mcfg.get("num_row_kv_heads", None)
            num_col_kv_heads = mcfg.get("num_col_kv_heads", None)
            bias = mcfg.get("bias", False)
        else:
            num_layers = cfg.model.num_layers
            num_heads = cfg.model.num_heads
            embedding_dim = cfg.model.num_hiddens
            num_row_kv_heads = cfg.model.num_row_kv_heads
            num_col_kv_heads = cfg.model.num_col_kv_heads
            bias = cfg.model.bias

        is_causal = cfg.model.get("is_causal", False)
        if is_causal:
            warn_rank_0("using causal model")

        dual_path = cfg.model.get("dual_path", False)
        sliding_window_size = cfg.model.get("sliding_window_size", None)
        tokenizer = build_tokenizer(cfg)
        return {
            "tokenizer": tokenizer,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_row_kv_heads": num_row_kv_heads,
            "num_col_kv_heads": num_col_kv_heads,
            "sliding_window_size": sliding_window_size,
            "is_causal": is_causal,
            "bias": bias,
            "eps": cfg.model.eps,
            "droppath": cfg.model.droppath,
            "dual_path": dual_path
        }

    @configurable
    def __init__(self,
                 tokenizer=None,
                 embedding_dim=512,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 num_row_kv_heads: Optional[int] = None,
                 num_col_kv_heads: Optional[int] = None,
                 ffn_dim_multiplier: Optional[float] = None,
                 sliding_window_size: Optional[int] = None,
                 bias: bool = False,
                 droppath: float = 0.0,
                 is_causal: bool = False,
                 dual_path: bool = False,
                 with_column_attention: bool = True,
                 eps: float = 1e-5,
                 include_head=True):
        super().__init__()
        self.vocab_size = len(tokenizer)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = self.embedding_dim // self.num_heads
        self.eps = eps
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.padding_idx = tokenizer.pad
        else:
            self.padding_idx = None

        self.embedding = Embedding(self.vocab_size, embedding_dim, padding_idx=self.padding_idx)
        self.is_causal = is_causal
        self.layers = nn.ModuleList([
            USMBlock(
                self.embedding_dim,
                self.num_heads,
                is_causal=is_causal,
                num_row_kv_heads=num_row_kv_heads,
                num_col_kv_heads=num_col_kv_heads,
                ffn_dim_multiplier=ffn_dim_multiplier,
                sliding_window_size=sliding_window_size,
                with_column_attention=with_column_attention,
                droppath=droppath,
                dual_path=dual_path,
                eps=self.eps) for _ in
            range(self.num_layers)
        ])
        self.norm_final = RMSNorm(self.embedding_dim, eps=self.eps)
        self.include_head = include_head
        self.apply(self._init_weights)
        self.lm_head = nn.Linear(embedding_dim, self.vocab_size, bias=bias) if self.include_head else None
        self.activation_checkpoint = False
        self.activation_checkpoint_fn = deepspeed.checkpointing.checkpoint

    def enable_activation_checkpoint(self, enabled=True):
        self.activation_checkpoint = enabled

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))

    def _forward_embedding_impl(self, token_ids, mask=None, **kwargs):
        return self.embedding(token_ids, mask=mask)

    def _forward_transformer_imp(self,
                                 x,
                                 mask: Optional[torch.Tensor] = None,
                                 return_attn_weight: bool = False,
                                 chunk_size: int = None,
                                 repr_layers=()
                                 ):
        attn_weights = []
        representations = {}
        for layer_id, block in enumerate(self.layers):
            if self.activation_checkpoint and torch.is_grad_enabled():
                x, attn_weight = self.activation_checkpoint_fn(block, x, mask, return_attn_weight)
            else:
                x, attn_weight = block(x,
                                       mask=mask,
                                       chunk_size=chunk_size,
                                       return_attn_weight=return_attn_weight)
            if return_attn_weight:
                attn_weights.append(attn_weight)

            if layer_id + 1 in repr_layers:
                representations[layer_id + 1] = x

        return x, attn_weights, representations

    def _forward_head_impl(self, inputs, mask: torch.Tensor = None):
        """
        Args:
            inputs: [*, seq_len, dim]
            mask: [*, seq_len]

        Returns:

        """
        x = inputs
        if mask is not None:
            x = x[mask]

        x = self.norm_final(x)
        if self.lm_head is not None:
            x = self.lm_head(x)

        if mask is not None:
            output = x.new_zeros(*inputs.shape[:-1], x.shape[-1])
            output[mask] = x
            x = output

        return x

    def _forward_impl(self,
                      token_ids,
                      repr_layers,
                      mask=None,
                      return_attn_weight=False):
        emb = self._forward_embedding_impl(token_ids, mask=mask)  # [bs, seq_len, hidden_dim]
        x, attn_weights, representations = self._forward_transformer_imp(emb,
                                                                         mask=mask,
                                                                         repr_layers=repr_layers,
                                                                         return_attn_weight=return_attn_weight)
        if 0 in repr_layers:
            representations[0] = emb

        logits = self._forward_head_impl(x, mask=mask)
        return logits, representations, attn_weights

    def forward(self,
                token_ids: torch.Tensor,
                repr_layers=(),
                return_attn_weight: bool = False
                ):
        """
        Args:
            token_ids: [bs, num_seq, seq_len], input mas token ids
        """
        is_single_seq = token_ids.dim() == 2
        if is_single_seq:
            # single seq
            token_ids = token_ids[:, None]

        padding_mask = None
        if self.padding_idx is not None:
            padding_mask = token_ids.eq(self.padding_idx)  # B, row, col
            if not padding_mask.any():
                padding_mask = None  # [bs, row, col]

        mask = None if padding_mask is None else ~padding_mask
        repr_layers = [self.num_layers + 1 + l if l < 0 else l for l in repr_layers]
        repr_layers = list(set(repr_layers))

        logits, representations, attn_weights = self._forward_impl(token_ids,
                                                                   mask=mask,
                                                                   repr_layers=repr_layers,
                                                                   return_attn_weight=return_attn_weight)
        outputs = {"logits": logits,
                   "representations": representations,
                   "mask": mask}

        if is_single_seq:
            for name in ["logits", "mask"]:
                outputs[name] = outputs[name].squeeze(dim=1) if outputs[name] is not None else None

        if return_attn_weight:
            outputs["attentions"] = torch.stack(attn_weights, dim=2)

        return outputs


class USMClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self,
                 dim,
                 num_classes=2,
                 dropout=0.0,
                 activation="tanh"):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_proj = nn.Linear(dim, num_classes)
        self.act = nn.Tanh() if activation == "tanh" else nn.ReLU()

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


@MODEL_REGISTRY.register()
class USMClassifier(USM):

    @classmethod
    def from_config(cls, cfg):
        configs = super().from_config(cfg)
        return {
            "num_classes": cfg.model.num_classes,
            "pooler": cfg.model.pooler,
            **configs
        }

    @configurable
    def __init__(self,
                 num_classes: int,
                 pooler=None,
                 head_name="plain",
                 include_head=False,
                 **kwargs):

        super(USMClassifier, self).__init__(
            include_head=include_head,
            **kwargs
        )
        self.pooler = pooler.lower() if pooler is not None else "cls"
        if self.is_causal:
            self.pooler = "last_pad"

        assert self.pooler in ("avg", "cls", "last_pad")
        if head_name == "plain":
            self.classifier = nn.Linear(self.embedding_dim, num_classes)
        else:
            self.classifier = USMClassificationHead(self.embedding_dim, num_classes)

    def forward(self, token_ids: torch.Tensor, **kwargs):
        is_single_seq = token_ids.dim() == 2
        if is_single_seq:
            token_ids = token_ids[:, None]

        padding_mask = None
        if self.padding_idx is not None:
            padding_mask = token_ids.eq(self.padding_idx)  # bs, num_seqs, seq_len
            if not padding_mask.any():
                padding_mask = None  # [bs, num_seqs, seq_len]

        mask = None if padding_mask is None else ~padding_mask  # bs, num_seqs, seq_len
        emb = self._forward_embedding_impl(token_ids, mask=mask, **kwargs)  # [bs, seq_len, dim]
        hidden, _, _ = self._forward_transformer_imp(emb, mask=mask)  # [bs, num_seqs, seq_len, dim]
        if mask is None:
            mask = torch.ones_like(token_ids, dtype=torch.bool)  # bs, num_seqs, seq_len

        if self.pooler == "avg":
            hidden = masked_mean(mask[:, 0].unsqueeze(dim=-1), hidden[:, 0].float(), dim=1).type_as(hidden)
        elif self.pooler == "last_pad":
            batch_size = token_ids.size(0)
            sequence_lengths = torch.ne(token_ids[:, 0], self.padding_idx).sum(dim=-1) - 1
            hidden = hidden[torch.arange(batch_size, device=hidden.device), 0, sequence_lengths]
        else:
            hidden = hidden[:, 0, 0]

        x = self.norm_final(hidden)  # [bs, dim]
        return {"logits": self.classifier(x)}


@MODEL_REGISTRY.register()
class USMTokenClassifier(USMClassifier):

    def forward(self, token_ids: torch.Tensor, **kwargs):
        is_single_seq = token_ids.dim() == 2
        if is_single_seq:
            token_ids = token_ids[:, None]

        padding_mask = None
        if self.padding_idx is not None:
            padding_mask = token_ids.eq(self.padding_idx)  # bs, num_seqs, seq_len
            if not padding_mask.any():
                padding_mask = None  # [bs, num_seqs, seq_len]

        mask = None if padding_mask is None else ~padding_mask  # bs, num_seqs, seq_len
        emb = self._forward_embedding_impl(token_ids)  # [bs, seq_len, dim]
        hidden, _, _ = self._forward_transformer_imp(emb, mask=mask)  # [bs, num_seqs, seq_len, dim]
        if mask is None:
            mask = torch.ones_like(token_ids, dtype=torch.bool)  # bs, num_seqs, seq_len

        if self.pooler == "avg":
            hidden = masked_mean(mask, hidden, dim=1)
        else:
            hidden = hidden[:, 0]

        x = self.norm_final(hidden)  # [bs, dim]
        return {"logits": self.classifier(x)}
