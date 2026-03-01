# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
from tgnn.config import configurable
from tgnn.model.build import MODEL_REGISTRY
from tgnn.model.layer import Linear, Embedding
from tgnn.utils import ModelOutput
from tgnn.utils.tensor import masked_mean
from .usm import USM, USMClassificationHead


@MODEL_REGISTRY.register()
class VariantCallingUSM(USM):

    @classmethod
    def from_config(cls, cfg):
        config = super().from_config(cfg)
        return {
            **config,
            "profile_dim": cfg.model.get("profile_dim", 38),
            "num_allele_types": getattr(cfg.model, "num_allele_types", 24),
        }

    @configurable
    def __init__(self,
                 *args,
                 num_allele_types: int = 24,
                 profile_dim: int = 38,
                 **kwargs):
        super(VariantCallingUSM, self).__init__(*args, **kwargs, include_head=False)
        assert not self.is_causal, "variant calling not support casual model"
        self.strand_embedding = Embedding(4, self.embedding_dim)
        nn.init.zeros_(self.strand_embedding.weight)
        self.quality_embedding = Linear(2, self.embedding_dim, init="zeros")

        self.profile_dim = profile_dim
        self.profile_embedding = Linear(self.profile_dim, self.embedding_dim, init="zeros")

        self.num_at_types = num_allele_types
        self.at_cls = USMClassificationHead(self.embedding_dim, self.num_at_types)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _forward_embedding_impl(self,
                                msa_token_ids,
                                ins_msa_token_ids: torch.Tensor = None,
                                strand_ids: torch.Tensor = None,
                                qualities: torch.Tensor = None,
                                profiles: torch.Tensor = None,
                                mask=None):

        emb = self.embedding(msa_token_ids, mask=mask)
        if ins_msa_token_ids is not None:
            emb = (emb + self.embedding(ins_msa_token_ids, mask=mask)) * 0.5

        if profiles is not None:
            profile_emb = self.profile_embedding(profiles.to(self.dtype))
            if mask is not None:
                profile_emb = profile_emb * mask[:, 0].unsqueeze(-1)
            emb[:, 0] = emb[:, 0] + profile_emb

        if strand_ids is not None:
            emb = emb + self.strand_embedding(strand_ids, mask=mask)

        if qualities is not None:
            emb = emb + self.quality_embedding(qualities.to(self.dtype))

        return emb

    def _forward_head_impl(self, hidden: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            hidden: [*, num_seqs, seq_len, dim]

        Returns:
            output: [*, num_classes]
        """
        hidden = self.norm_final(hidden)  # [*, num_seqs, seq_len, dim]
        x = masked_mean(mask.unsqueeze(dim=-1) if mask is not None else None, hidden.float(),
                        dim=(1, 2)).type_as(hidden)
        output = ModelOutput(at=self.at_cls(x))

        return output

    def forward(self,
                msa_token_ids: torch.Tensor,
                ins_msa_token_ids: torch.Tensor = None,
                strand_ids: torch.Tensor = None,
                qualities: torch.Tensor = None,
                profiles: torch.Tensor = None):
        padding_mask = None
        if self.padding_idx is not None:
            padding_mask = msa_token_ids.eq(self.padding_idx)  # B, row, col
            if not padding_mask.any():
                padding_mask = None  # [bs, row, col]

        mask = None if padding_mask is None else ~padding_mask
        emb = self._forward_embedding_impl(msa_token_ids,
                                           ins_msa_token_ids,
                                           strand_ids=strand_ids,
                                           qualities=qualities,
                                           profiles=profiles,
                                           mask=mask)  # [bs, seq_len, hidden_dim]
        hidden, *_ = self._forward_transformer_imp(emb, mask=mask)
        outputs = self._forward_head_impl(hidden, mask=mask)

        return outputs
