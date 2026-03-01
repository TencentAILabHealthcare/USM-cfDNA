# Copyright (c) 2024, Tencent Inc. All rights reserved.

from .base_evaluator import DatasetEvaluator, LamadaEvaluator
from .build import build_evaluator, EVALUATOR_REGISTRY
from .confusion_matrix import MultiClassConfusionMatrixEvaluator
from .variant_evaluator import VariantEvaluator
from .methylation_evaluator import MethylationEvaluator
from .cancer_evaluator import MCEDEvaluator