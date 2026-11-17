"""Exhaustive sweep of the mask post-processing (1,120 combinations), with selection and reporting on disjoint data.

Axes: head (252x336 / 378x504) x cut (p > 0.3..0.7, hysteresis high {0.5,0.6,0.7} x low {0.1,0.2,0.3})
      x erosion 0-4 px x blend range {0-4, 2-6, 2-8, 4-8, 4-12, 6-12, 6-16, 8-20 px}.
Split A = every 8th test image (choose), split B = the images 4 later (report). Prints the Pareto set on A
(no combination is better on transparent, surround and background at once) and re-measures the candidates on B.

    python scripts/sweep_mask.py --out results/mask_sweep.json
"""
import argparse
import json

from _common import setup, add_common_args

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tdc import DA_SIZE_HI
from tdc.data import TransCG320, to_device
from tdc.gate import erode, hysteresis
from tdc.metrics import regions
from tdc.models import DAFeatures, DepthNet, MaskHead, load_state

CUTS = [(f"p>{t:.1f}", ("p", t, None)) for t in (0.3, 0.4, 0.5, 0.6, 0.7)] + \
       [(f"hyst {hi:.1f}/{lo:.1f}", ("h", hi, lo)) for hi in (0.5, 0.6, 0.7) for lo in (0.1, 0.2, 0.3)]
EROS = [0, 1, 2, 3, 4]
BLENDS = [(0, 4), (2, 6), (2, 8), (4, 8), (4, 12), (6, 12), (6, 16), (8, 20)]
KEYS = ["transparent", "boundary", "surround", "background", "iou", "D435", "L515"]
CURRENT = {"head": ("hyst 0.5/0.1", 2, (4, 12)), "head_hi": ("hyst 0.5/0.1", 2, (4, 12))}


def distance(m):
    """ceil(Euclidean distance to the mask), 0 inside; identical to tdc.gate.distance_to_mask without the radius cap"""
    out = []
    for x in (~m).cpu().numpy().astype(np.uint8):
        out.append(np.ceil(cv2.distanceTransform(x, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) - 1e-4) if x.any() else np.full(x.shape, 99.0, np.float32))
    return torch.from_numpy(np.stack(out)).float().to(m.device)


def run(idx, ds, models, combos, dev, workers):
    S = {}
    with torch.no_grad():
        for b in DataLoader(Subset(ds, idx), 16, shuffle=False, num_workers=workers, pin_memory=True):
            b = to_device(b, dev); rgb, depth, gt, gm = b["rgb"], b["depth"], b["gt"], b["mask"]; cams = np.array(b["cam"].tolist())
            fe = models["feats"](rgb); D = models["depth"](rgb, depth, fe)
            P = {"head": torch.sigmoid(models["head"](rgb, depth, fe)).float(),
                 "head_hi": torch.sigmoid(models["head_hi"](rgb, depth, models["feats_hi"](rgb))).float()}
            reg = regions(gm); rg = (gt >= 0.3) & (gt <= 1.5)
            R = torch.stack([gm & (gt > 0), reg[:, 0] & rg, reg[:, 3] & rg, reg[:, 4] & rg]); cnt = R.sum((2, 3)).float()
            for hn in P:
                for cn, (k, t1, t2) in CUTS:
                    m0 = (P[hn] > t1) if k == "p" else hysteresis(P[hn], t1, t2)
                    for e in EROS:
                        if combos is not None and not any(c[:3] == (hn, cn, e) for c in combos):
                            continue
                        m = erode(m0, e); dd = distance(m)
                        iou = ((m & gm).sum((1, 2)).float() / (m | gm).sum((1, 2)).float().clamp(min=1)).cpu().numpy()
                        for lo, hi in BLENDS:
                            key = (hn, cn, e, (lo, hi))
                            if combos is not None and key not in combos:
                                continue
                            w = torch.where(depth > 0, ((hi - dd) / (hi - lo)).clamp(0, 1), torch.ones_like(dd))
                            e2 = (w * D + (1 - w) * depth - gt) ** 2
                            r = torch.sqrt((e2[None] * R).sum((2, 3)) / cnt.clamp(min=1)) * 1000
                            r = torch.where(cnt > 0, r, torch.nan).cpu().numpy()
                            rows = [r[0], r[1], r[2], r[3], iou, np.where(cams == 1, r[0], np.nan), np.where(cams == 2, r[0], np.nan)]
                            acc = S.setdefault(key, np.zeros((len(KEYS), 2)))
                            for j, x in enumerate(rows):
                                ok = ~np.isnan(x); acc[j, 0] += x[ok].sum(); acc[j, 1] += ok.sum()
    return {k: {kk: float(v[j, 0] / max(v[j, 1], 1)) for j, kk in enumerate(KEYS)} for k, v in S.items()}


def pareto(res, keys=("transparent", "surround", "background")):
    return {k for k in res if not any(all(res[j][q] <= res[k][q] for q in keys) and any(res[j][q] < res[k][q] for q in keys)
                                      for j in res if j != k)}


def name(k):
    return f"{k[0]} | {k[1]} | erode {k[2]} | blend {k[3][0]}-{k[3][1]}"


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--out", default="results/mask_sweep.json")
    a = ap.parse_args()
    dev, _ = setup(0)
    feats = DAFeatures().to(dev)

    class _B:
        backbone = feats.backbone
    models = {"feats": feats, "feats_hi": DAFeatures(size=DA_SIZE_HI, model=_B()).to(dev),
              "depth": load_state(DepthNet(True), f"{a.ckpt}/depth_da.pth").to(dev).eval(),
              "head": load_state(MaskHead(), f"{a.ckpt}/mask_head.pth").to(dev).eval(),
              "head_hi": load_state(MaskHead(), f"{a.ckpt}/mask_head_hi.pth").to(dev).eval()}
    ds = TransCG320(a.data, "test")
    A, B = list(range(0, len(ds), 8)), list(range(4, len(ds), 8))
    print(f"A {len(A)} images (select), B {len(B)} images (report)")
    RA = run(A, ds, models, None, dev, a.workers)
    PA = pareto(RA)
    cand = {}
    for hn, cur in CURRENT.items():
        now = (hn, *cur); cand[f"{hn}: current"] = now
        sub = {k: v for k, v in RA.items() if k[0] == hn}
        cand[f"{hn}: lowest transparent"] = min(sub, key=lambda k: sub[k]["transparent"])
        ok = [k for k in sub if sub[k]["background"] <= RA[now]["background"] and sub[k]["surround"] <= RA[now]["surround"]]
        cand[f"{hn}: lowest transparent, background & surround not worse"] = min(ok, key=lambda k: sub[k]["transparent"])
        ok = [k for k in sub if sub[k]["transparent"] <= RA[now]["transparent"] + 0.05]
        cand[f"{hn}: lowest surround, transparent +0.05"] = min(ok, key=lambda k: sub[k]["surround"])
        cand[f"{hn}: lowest background, transparent +0.05"] = min(ok, key=lambda k: sub[k]["background"])
    RB = run(B, ds, models, set(cand.values()), dev, a.workers)
    print(f"\nPareto set on A: {len(PA)} of {len(RA)} combinations. Candidates re-measured on B (mm):")
    print(f"{'candidate':55s}{'combination':48s}" + "".join(f"{k[:6]:>8s}" for k in KEYS[:4]))
    for cn, k in cand.items():
        print(f"{cn:55s}{name(k):48s}" + "".join(f"{RB[k][q]:8.2f}" for q in KEYS[:4]))
    json.dump({"A": {name(k): v for k, v in RA.items()}, "pareto_A": [name(k) for k in PA],
               "candidates_B": {cn: {"combination": name(k), **RB[k]} for cn, k in cand.items()}}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
