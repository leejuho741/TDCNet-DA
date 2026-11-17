"""Metrics.

* StandardMetrics: TransCG paper protocol. Inside the GT transparent mask (GT > 0.01 m), per image, then averaged:
  RMSE, REL, MAE (mm / %) and delta 1.05 / 1.10 / 1.25 (%).
* RegionMetrics: per-image RMSE in regions defined around the GT mask (320x240 pixels, GT in 0.3-1.5 m):
  boundary = 3 px inside + 3 px outside the mask edge, surround = 3-15 px outside, background = > 15 px outside.
  'transparent' is the RMSE inside the mask computed at 256x144 (the official evaluation size).
  Background error is what the gate fixes: methods without a gate rewrite opaque surfaces too.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import CAM_NAME

STD_KEYS = ["rmse", "rel", "mae", "d105", "d110", "d125"]
REGION_KEYS = ["transparent", "d105", "boundary", "inner", "outer", "surround", "background", "iou"]


class StandardMetrics:
    def __init__(self, cam_names=CAM_NAME):
        self.d, self.cam_names = {}, cam_names

    def add(self, name, out, gt, gt_mask, cams):
        for j in range(out.shape[0]):
            sel = gt_mask[j] & (gt[j] > 0.01)
            if int(sel.sum()) < 10:
                continue
            o, g = out[j][sel].float(), gt[j][sel].float()
            e = o - g
            r = torch.maximum(o.clamp(min=1e-6) / g, g / o.clamp(min=1e-6))
            v = [float(e.pow(2).mean().sqrt()) * 1000, float((e.abs() / g).mean()) * 100, float(e.abs().mean()) * 1000,
                 float((r < 1.05).float().mean()) * 100, float((r < 1.10).float().mean()) * 100, float((r < 1.25).float().mean()) * 100]
            for k in ("all", self.cam_names[cams[j]]):
                self.d.setdefault(name, {}).setdefault(k, []).append(v)

    def summary(self, name, split="all"):
        v = self.d.get(name, {}).get(split)
        return dict(zip(STD_KEYS, np.mean(v, 0).tolist()), n=len(v)) if v else None


_ELL = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def regions(gt_mask):
    """(N, 5, H, W) bool: boundary, inner band, outer band, surround, background."""
    out = []
    for m in gt_mask.cpu().numpy().astype(np.uint8):
        b3, n15 = cv2.dilate(m, _ELL(3)) > 0, cv2.dilate(m, _ELL(15)) > 0
        e3 = cv2.erode(m, _ELL(3)) > 0
        mb = m > 0
        inn, outb = mb & ~e3, b3 & ~mb
        out.append(np.stack([inn | outb, inn, outb, n15 & ~b3, ~n15]))
    return torch.from_numpy(np.stack(out)).to(gt_mask.device)


class RegionMetrics:
    def __init__(self, cam_names=CAM_NAME):
        self.d, self.cam_names = {}, cam_names

    def _add(self, name, key, vals, cams):
        for v, c in zip(vals, cams):
            if v == v:
                dd = self.d.setdefault(name, {}).setdefault(key, {"all": []})
                dd["all"].append(v)
                dd.setdefault(self.cam_names[c], []).append(v)

    def add(self, name, out, gt, gt_mask, cams, reg=None, pred_mask=None):
        reg = regions(gt_mask) if reg is None else reg
        e2 = (out - gt) ** 2
        rs = lambda x: F.interpolate(x[:, None].float(), size=(144, 256), mode="nearest")[:, 0]
        o_, g_, m_ = rs(out), rs(gt), rs(gt_mask) > 0.5
        mm = m_ & (g_ > 0)
        c_ = mm.sum((1, 2)).float()
        ok = c_ > 0
        r256 = torch.sqrt((((o_ - g_) ** 2) * mm).sum((1, 2)) / c_.clamp(min=1)) * 1000
        q = torch.maximum(o_ / g_.clamp(min=1e-8), g_ / o_.clamp(min=1e-8))
        d256 = ((q < 1.05) & mm).sum((1, 2)).float() / c_.clamp(min=1) * 100
        self._add(name, "transparent", torch.where(ok, r256, torch.nan).tolist(), cams)
        self._add(name, "d105", torch.where(ok, d256, torch.nan).tolist(), cams)
        rg = (gt >= 0.3) & (gt <= 1.5)
        for j, rn in enumerate(["boundary", "inner", "outer", "surround", "background"]):
            s = reg[:, j] & rg
            cn = s.sum((1, 2)).float()
            self._add(name, rn, torch.where(cn > 0, torch.sqrt((e2 * s).sum((1, 2)) / cn.clamp(min=1)) * 1000, torch.nan).tolist(), cams)
        if pred_mask is not None:
            inter = (pred_mask & gt_mask).sum((1, 2)).float()
            uni = (pred_mask | gt_mask).sum((1, 2)).float()
            self._add(name, "iou", torch.where(uni > 0, inter / uni.clamp(min=1), torch.nan).tolist(), cams)

    def summary(self, name, split="all"):
        out = {}
        for k in REGION_KEYS:
            v = self.d.get(name, {}).get(k, {}).get(split, [])
            out[k] = float(np.mean(v)) if v else float("nan")
        return out
