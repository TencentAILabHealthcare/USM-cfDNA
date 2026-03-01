# Copyright (c) 2025, Tencent Inc. All rights reserved.
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from tgnn.model.utils.init_weights import INIT_METHOD_REGISTRY


class Embedding(nn.Embedding):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 init: Union[str, Callable, None] = None,
                 device=None,
                 dtype=None,
                 ):
        self.init = init
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                         device=device, dtype=dtype)

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input: [*, seq_len]
            mask: [*, seq_len]

        Returns:
            [*, seq_len]
        """
        if mask is None or input.ndim == 1:
            return super().forward(input)

        output = torch.zeros(*input.shape,
                             self.embedding_dim,
                             dtype=self.dtype, device=input.device)
        output[mask] = super().forward(input[mask])

        return output

    def reset_parameters(self) -> None:
        init = self.init
        assert isinstance(init, (str, Callable, type(None)))
        if isinstance(init, str):
            INIT_METHOD_REGISTRY[init](self.weight)
        elif isinstance(init, Callable):
            return init(self.weight)
        else:
            super().reset_parameters()
