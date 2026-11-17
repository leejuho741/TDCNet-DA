"""Shared helpers for the scripts (path setup, device, seeding, logging)."""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")   # official third-party checkpoints contain NumPy values

import numpy as np  # noqa: E402
import torch  # noqa: E402


def setup(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if not hasattr(np, "bool"):           # old third-party code uses np.bool
        np.bool = bool
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    return dev, amp


def add_common_args(p):
    p.add_argument("--data", default=os.environ.get("TRANSCG_320", "data/transcg_320"), help="TransCG resized to 320x240")
    p.add_argument("--ckpt", default="checkpoints", help="checkpoint folder")
    p.add_argument("--workers", type=int, default=8)
    return p
