# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.

import psutil


def get_cpu_cores(logical=True):
    return psutil.cpu_count(logical=logical)
