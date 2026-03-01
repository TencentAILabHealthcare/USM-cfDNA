# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.

from .build import TOKENIZER_REGISTRY, build_tokenizer, create_tokenizer
from .alphabet import Alphabet
from .sentencepiece import SentencePieceTokenizer, SpeicalMappingSentencePieceTokenizer
from .utils import fit_vocab_size_to_dtype