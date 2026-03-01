# Copyright (c) 2025, Tencent Inc. All rights reserved.
import torch
import torch.nn as nn

from tgnn.config import configurable
from tgnn.model.build import MODEL_REGISTRY
from tgnn.model.layer import Linear, Embedding
from .usm import USMClassifier


@MODEL_REGISTRY.register()
class MethylationUSM(USMClassifier):

    @classmethod
    def from_config(cls, cfg):
        configs = super().from_config(cfg)
        return {
            **configs,
            "profile_dim": cfg.model.get("profile_dim", 5)
        }

    @configurable
    def __init__(self, *args, profile_dim=5, **kwargs):
        super().__init__(*args, **kwargs, head_name="usm", include_head=True)
        self.strand_embedding = Embedding(4, self.embedding_dim, padding_idx=3)
        nn.init.zeros_(self.strand_embedding.weight)
        self.qual_embedding = Linear(2, self.embedding_dim, init="zeros")
        self.profile_embedding = Linear(profile_dim, self.embedding_dim, init="zeros")

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
        """
        Args:
            msa_token_ids: [bs, num_seqs, seq_len]
            mask: [bs, num_seqs, seq_len]

        Returns:
            embedding: [bs, num_seqs, seq_len, dim]
        """
        emb = (self.embedding(msa_token_ids, mask=mask) + self.embedding(ins_msa_token_ids, mask=mask)) * 0.5
        if strand_ids is not None:
            emb = emb + self.strand_embedding(strand_ids, mask=mask)

        if qualities is not None:
            qual_emb = self.qual_embedding(qualities.to(self.dtype))
            if mask is not None:
                qual_emb = qual_emb * mask.unsqueeze(-1)

            emb = emb + qual_emb

        if profiles is not None:
            profile_emb = self.profile_embedding(profiles[:, None].to(self.dtype))
            if mask is not None:
                profile_emb = profile_emb * mask.unsqueeze(-1)
            emb = emb + profile_emb

        return emb

    def forward(self,
                msa_token_ids: torch.Tensor,
                ins_msa_token_ids: torch.Tensor = None,
                strand_ids: torch.Tensor = None,
                qualities: torch.Tensor = None,
                profiles: torch.Tensor = None
                ):
        return super().forward(msa_token_ids,
                               ins_msa_token_ids=ins_msa_token_ids,
                               strand_ids=strand_ids,
                               qualities=qualities,
                               profiles=profiles)
