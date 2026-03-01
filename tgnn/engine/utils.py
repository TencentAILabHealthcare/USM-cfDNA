# Copyright (c) 2024, Tencent Inc. All rights reserved.

import importlib
import logging
import os
import sys


def auto_register_settings(code_dir=".", moudle_list=None):
    """load moudle or python file from local

    Args:
        cade_dir: path, local project directory
        moudle_list: list[str], moudle name list
    """
    sys.path.append(code_dir)
    if moudle_list is None:
        moudle_list = ("config", "dataset", "criterion", "model", "trainer")

    for moudle in moudle_list:
        # load moudle directory or python file
        moudle_file = f"{code_dir}/{moudle}.py"
        moudle_dir = f"{code_dir}/{moudle}"
        if os.path.exists(moudle_file) or os.path.exists(moudle_dir):
            try:
                importlib.import_module(moudle)
                logging.warning(f"auto import moudle file: {moudle}")
            except Exception as e:
                logging.warning(f"can not import moudle file: {moudle}\n{e}")
