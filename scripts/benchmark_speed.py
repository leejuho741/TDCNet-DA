"""Latency of the full pipeline: batch 1, 320x240, fp16 autocast + CUDA graph, median of 3 x 200 runs.

    python scripts/benchmark_speed.py --modes fast accurate student

Reference (NVIDIA L4): fast 66-69 FPS, accurate 31 FPS, student 85 FPS, TDCNet alone 177-182 FPS.
Runs vary by about +-3 FPS; compare methods measured in the same run.
"""
import argparse

from _common import setup, add_common_args

import torch

from tdc.data import TransCG320
from tdc.models import DepthNet, load_state
from tdc.pipeline import build
from tdc.speed import latency_ms


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--modes", nargs="+", default=["tdcnet", "fast", "accurate", "student"])
    a = ap.parse_args()
    dev, _ = setup(0)
    s = TransCG320(a.data, "test")[0]
    rgb, depth = s["rgb"][None].to(dev).contiguous(), s["depth"][None].to(dev).contiguous()
    print(f"GPU {torch.cuda.get_device_name()}")
    for m in a.modes:
        if m == "tdcnet":
            net = load_state(DepthNet(False), f"{a.ckpt}/tdcnet_repro.pth").to(dev).eval()
            fn = lambda net=net: net(rgb, depth)
        else:
            model = build(m, a.ckpt, dev)
            fn = lambda model=model: model(rgb, depth)
        ms, how = latency_ms(fn)
        print(f"  {m:10s} {1000 / ms:6.0f} FPS  {ms:6.2f} ms  ({how})")


if __name__ == "__main__":
    main()
