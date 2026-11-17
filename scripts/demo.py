"""Run the model on one RGB-D pair and save a side-by-side picture (RGB | sensor | completed | mask).

    python scripts/demo.py --rgb rgb1.png --depth depth1.png --scale 1000 --mode fast --out demo.png

--scale: depth PNG units per metre (TransCG D435 1000, L515 4000). Inputs are resized to 320x240.
"""
import argparse

from _common import setup

import cv2
import numpy as np
import torch

from tdc import H, W
from tdc.pipeline import build


def colorize(d, lo, hi):
    x = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    c = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[d <= 0] = 0
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--scale", type=float, default=1000.0)
    ap.add_argument("--mode", choices=["fast", "accurate", "student"], default="fast")
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--out", default="demo.png")
    a = ap.parse_args()
    dev, _ = setup(0)
    rgb = cv2.resize(cv2.cvtColor(cv2.imread(a.rgb), cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dep = cv2.resize(cv2.imread(a.depth, cv2.IMREAD_ANYDEPTH).astype(np.float32) / a.scale, (W, H), interpolation=cv2.INTER_NEAREST)
    dep = np.minimum(dep, 1.5)
    model = build(a.mode, a.ckpt, dev)
    with torch.no_grad():
        out, aux = model(torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev), torch.from_numpy(dep)[None].to(dev), return_all=True)
    out = out[0].cpu().numpy(); m = aux["mask"][0].cpu().numpy()
    v = dep[dep > 0]
    lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0.3, 1.5)
    tiles = [cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), colorize(dep, lo, hi), colorize(out, lo, hi),
             cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]
    cv2.imwrite(a.out, np.concatenate(tiles, 1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
