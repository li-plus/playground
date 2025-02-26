"""
LINE_PROFILE=1 python3 line_prof.py
"""

import torch
from line_profiler import profile


@profile
def work():
    a = torch.randn(1024, 1024)
    a + 1
    a * 5
    a * a
    a @ a


work()
