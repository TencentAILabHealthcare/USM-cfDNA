# Copyright (c) 2025, Tencent Inc. All rights reserved.

import importlib
from pathlib import Path

current_dir = Path(__file__).parent
files = list(current_dir.glob("[!_]*.py"))

for path in files:
    moudle = path.stem
    moudle = f"tgnn.config.meta.{moudle}"
    importlib.import_module(moudle)