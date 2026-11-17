"""Evaluate on TransCG test (23,524 images) or ClearGrasp real-test (113 images, zero-shot).

    python scripts/evaluate.py --methods tdcnet depth_only fast accurate              # TransCG test, all images
    python scripts/evaluate.py --methods fast --every 40                              # quick check (588 images)
    python scripts/evaluate.py --dataset cleargrasp --cleargrasp data/cleargrasp --methods fast

Methods
  tdcnet      TDCNet reproduced with the official recipe (no gate)
  depth_only  TDCNet + DA injection, raw output (no mask head, no gate)
  fast        + mask head + gate                       (speed-oriented)
  accurate    + high-res mask head + flip TTA          (accuracy-oriented)
  student     distilled ResNet-18 instead of Depth Anything
  adapter_aug frozen TDCNet + strong-augmentation adapter, with gate
Baselines from other repositories (FDCT, TODE, ReMake) were evaluated with experiments/34_sota_full.py.
"""
import argparse
import json

from _common import setup, add_common_args

import torch
from torch.utils.data import DataLoader, Subset

from tdc.data import ClearGraspReal, TransCG320, cleargrasp_real_files, to_device
from tdc.gate import gate, mask_from_prob
from tdc.metrics import RegionMetrics, StandardMetrics, regions
from tdc.models import DAFeatures, DepthNet, MaskHead, load_state
from tdc.pipeline import build


def make_methods(names, ckpt, dev):
    M, shared = {}, {}

    def da():
        if "da" not in shared:
            shared["da"] = DAFeatures().to(dev)
        return shared["da"]

    for n in names:
        if n == "tdcnet":
            net = load_state(DepthNet(False), f"{ckpt}/tdcnet_repro.pth").to(dev).eval()
            M[n] = lambda rgb, d, net=net: (net(rgb, d), None)
        elif n == "depth_only":
            net = load_state(DepthNet(True), f"{ckpt}/depth_da.pth").to(dev).eval()
            M[n] = lambda rgb, d, net=net: (net(rgb, d, da()(rgb)), None)
        elif n == "adapter_aug":
            net = load_state(DepthNet(True), f"{ckpt}/adapter_aug.pth").to(dev).eval()
            head = load_state(MaskHead(), f"{ckpt}/mask_head.pth").to(dev).eval()

            def f(rgb, d, net=net, head=head):
                fe = da()(rgb); m = mask_from_prob(torch.sigmoid(head(rgb, d, fe)).float())
                return gate(net(rgb, d, fe), d, m), m
            M[n] = f
        elif n in ("fast", "accurate", "student"):
            model = build(n, ckpt, dev)

            def f(rgb, d, model=model):
                out, aux = model(rgb, d, return_all=True)
                return out, aux["mask"]
            M[n] = f
        else:
            raise ValueError(n)
    return M


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--dataset", choices=["transcg", "cleargrasp"], default="transcg")
    ap.add_argument("--cleargrasp", default="data/cleargrasp")
    ap.add_argument("--methods", nargs="+", default=["tdcnet", "depth_only", "fast", "accurate"])
    ap.add_argument("--every", type=int, default=1, help="use every k-th test image (1 = all)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default=None, help="write results JSON")
    a = ap.parse_args()
    dev, _ = setup(0)
    if a.dataset == "transcg":
        ds = TransCG320(a.data, "test")
        cams = {1: "D435", 2: "L515"}
    else:
        ds = ClearGraspReal(cleargrasp_real_files(a.cleargrasp, "real-test"))
        cams = {1: "D435", 2: "D415"}
    if a.every > 1:
        ds = Subset(ds, list(range(0, len(ds), a.every)))
    print(f"{a.dataset}: {len(ds)} images")
    M = make_methods(a.methods, a.ckpt, dev)
    std, reg = StandardMetrics(cams), RegionMetrics(cams)
    with torch.no_grad():
        for i, b in enumerate(DataLoader(ds, a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)):
            b = to_device(b, dev); c = b["cam"].tolist(); r = regions(b["mask"])
            for n, f in M.items():
                o, m = f(b["rgb"], b["depth"])
                std.add(n, o, b["gt"], b["mask"], c); reg.add(n, o, b["gt"], b["mask"], c, reg=r, pred_mask=m)
            if i % 200 == 0:
                print(f"  {min((i + 1) * a.bs, len(ds))}/{len(ds)}")
    res = {}
    print(f"\n{'method':12s}{'RMSE':>8s}{'d1.05':>8s}{'MAE':>8s}{'bound.':>8s}{'surr.':>8s}{'backgr.':>9s}"
          + "".join(f"{v:>8s}" for v in cams.values()) + f"{'IoU':>7s}")
    for n in M:
        s, g = std.summary(n), reg.summary(n)
        per_cam = {v: (std.summary(n, v) or {}).get("rmse", float("nan")) for v in cams.values()}
        res[n] = {"standard": s, "regions": g, "rmse_per_camera": per_cam}
        print(f"{n:12s}{s['rmse']:8.2f}{s['d105']:8.2f}{s['mae']:8.2f}{g['boundary']:8.2f}{g['surround']:8.2f}{g['background']:9.2f}"
              + "".join(f"{x:8.2f}" for x in per_cam.values()) + f"{g['iou']:7.3f}")
    print("RMSE / MAE / regions in mm; d1.05 in %. RMSE, d1.05, MAE: inside the GT transparent mask (TransCG protocol).")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
