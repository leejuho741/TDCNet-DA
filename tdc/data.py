"""Datasets.

TransCG is used at 320x240 (``scripts/prepare_transcg_320.py`` writes the same folder layout as the original
dataset, only resized). Depth PNGs are in camera units: D435 = 1 mm, L515 = 0.25 mm.
"""
import glob
import json
import os

import cv2
import numpy as np
import torch

from . import H, W

DEPTH_SCALE = {1: 1000.0, 2: 4000.0}      # camera id -> units per metre (1 = D435, 2 = L515)
CAM_NAME = {1: "D435", 2: "L515"}


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def intrinsics_table(data_dir, device="cpu"):
    """Rows: [unused, D435, L515] x [fx, fy, cx, cy] at 1280x720, used for surface normals in the loss."""
    k = {c: np.load(f"{data_dir}/camera_intrinsics/{c}-camIntrinsics-{CAM_NAME[c]}.npy") for c in (1, 2)}
    return torch.tensor([[1.0, 1.0, 0.0, 0.0]] + [[k[c][0, 0], k[c][1, 1], k[c][0, 2], k[c][1, 2]] for c in (1, 2)], device=device)


class TransCG320(torch.utils.data.Dataset):
    """TransCG at 320x240.
    train: depth > 10 m set to 0, random flips (both axes) + optional colour jitter / noise.
    test:  depth and GT clipped at 1.5 m (as in the official evaluation).
    Returns rgb (3,H,W) [0,1], depth (H,W) m, gt (H,W) m, mask (H,W) bool, cam (1 D435 / 2 L515)."""

    def __init__(self, data_dir, split="test", train=False, color=True):
        self.train, self.color, self.items = train, color, []
        for d in sorted(glob.glob(f"{data_dir}/scene*"), key=lambda p: int(os.path.basename(p)[5:])):
            meta = json.load(open(f"{d}/metadata.json"))
            if meta.get("split") != split:
                continue
            for cam, key in [(1, "D435_valid_perspective_list"), (2, "L515_valid_perspective_list")]:
                self.items += [(f"{d}/{v}", cam) for v in meta[key] if os.path.exists(f"{d}/{v}/rgb{cam}.png")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, cam = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(f"{d}/rgb{cam}.png"), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dep = cv2.imread(f"{d}/depth{cam}.png", cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE[cam]
        gt = cv2.imread(f"{d}/depth{cam}-gt.png", cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE[cam]
        for x in (dep, gt):
            x[~np.isfinite(x) | (x < 0)] = 0
        if self.train:
            dep[dep > 10] = 0
            gt[gt > 10] = 0
        else:
            dep = np.minimum(dep, 1.5)
            gt = np.minimum(gt, 1.5)
        mask = cv2.imread(f"{d}/depth{cam}-gt-mask.png", cv2.IMREAD_GRAYSCALE) > 0
        if self.train:
            if self.color and np.random.rand() < 0.8:
                rgb = rgb * np.random.uniform(0.8, 1.2)
                mu = rgb.mean()
                rgb = (rgb - mu) * np.random.uniform(0.8, 1.2) + mu
                rgb = rgb * np.random.uniform(0.95, 1.05, size=3).astype(np.float32)
                if np.random.rand() < 0.5:
                    rgb = rgb + np.random.randn(*rgb.shape).astype(np.float32) * np.random.uniform(0, 0.03)
                rgb = np.clip(rgb, 0, 1).astype(np.float32)
            for ax in (0, 1):
                if np.random.rand() > 0.5:
                    rgb, dep, gt, mask = (np.flip(x, ax) for x in (rgb, dep, gt, mask))
            rgb, dep, gt, mask = (np.ascontiguousarray(x) for x in (rgb, dep, gt, mask))
        return dict(rgb=torch.from_numpy(rgb).permute(2, 0, 1), depth=torch.from_numpy(dep), gt=torch.from_numpy(gt),
                    mask=torch.from_numpy(mask), cam=cam)


def read_exr(path):
    x = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if x is None:
        import OpenEXR, Imath
        f = OpenEXR.InputFile(path)
        dw = f.header()["dataWindow"]
        size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        ch = list(f.header()["channels"].keys())[0]
        x = np.frombuffer(f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(size)
    if x.ndim == 3:
        x = x[..., 0]
    x = x.astype(np.float32)
    x[~np.isfinite(x) | (x < 0)] = 0
    return x


def cleargrasp_real_files(root, split):
    """split: 'real-val' (173 images, used for adaptation) or 'real-test' (113 images, evaluation only)."""
    return sorted(f for f in glob.glob(f"{root}/**/*-transparent-rgb-img.jpg", recursive=True) if f"/{split}/" in f)


class ClearGraspReal(torch.utils.data.Dataset):
    """ClearGrasp real images resized to 320x240, depth clipped at 1.5 m. cam: 1 = D435, 2 = D415."""

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        pre = p[: -len("-transparent-rgb-img.jpg")]
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        dep = read_exr(pre + "-transparent-depth-img.exr")
        gt = read_exr(pre + "-opaque-depth-img.exr")
        mk = cv2.imread(pre + "-mask.png", cv2.IMREAD_GRAYSCALE)
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        dep, gt = (np.minimum(cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST), 1.5) for x in (dep, gt))
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST) > 127
        cam = 1 if "d435" in p.lower() else 2
        return dict(rgb=torch.from_numpy(rgb).permute(2, 0, 1), depth=torch.from_numpy(dep), gt=torch.from_numpy(gt),
                    mask=torch.from_numpy(mk), cam=cam)
