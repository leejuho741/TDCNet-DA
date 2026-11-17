"""Strong-augmentation adapter ("+ strong data augmentation" row of the generalization table).

TDCNet (reproduced weights) is frozen; only the zero-initialised DA injection projections are trained, 3 epochs,
AdamW 1e-3 (wd 1e-4), 300-step warm-up + cosine, grad clip 1.0, with sensor-failure augmentation:
  * zoom 1.0-1.6x + crop (p 0.5)
  * transparent region: random holes (D435-like) or values from behind the object (L515-like), p 0.25 each
  * small holes in the background (p 0.3), slight blur (p 0.2)
Slightly worse on TransCG than the from-scratch model, but adapts better to a new domain (see README).

    python scripts/train_adapter.py --init checkpoints/tdcnet_repro.pth --out checkpoints/adapter_aug.pth
"""
import argparse
import math
import random
import time

from _common import setup, add_common_args

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tdc import H, W
from tdc.data import TransCG320, intrinsics_table, to_device
from tdc.losses import tdcnet_loss
from tdc.models import DAFeatures, DepthNet


def smooth_noise(n, lo_h, lo_w, dev):
    return F.interpolate(torch.rand(n, 1, lo_h, lo_w, device=dev), size=(H, W), mode="bilinear", align_corners=False)[:, 0]


def sensor_failure_aug(rgb, depth, gt, gm, rng):
    dev = rgb.device
    if rng.random() < 0.5:                                        # zoom + crop
        sc = rng.uniform(1.0, 1.6); h2, w2 = int(H * sc), int(W * sc)
        y0, x0 = rng.randint(0, h2 - H), rng.randint(0, w2 - W)
        rgb = F.interpolate(rgb, size=(h2, w2), mode="bilinear", align_corners=False)[..., y0:y0 + H, x0:x0 + W]
        up = lambda x: F.interpolate(x[:, None].float(), size=(h2, w2), mode="nearest")[:, 0][..., y0:y0 + H, x0:x0 + W]
        depth, gt, gm = up(depth), up(gt), up(gm) > 0.5
    depth = depth.clone()
    far = F.max_pool2d(depth[:, None], 15, 1, 7)[:, 0]            # value behind the object
    for j in range(rgb.shape[0]):
        r = rng.random()
        if r < 0.25:                                              # holes in the transparent region
            depth[j][gm[j] & (smooth_noise(1, 12, 16, dev)[0] > rng.uniform(0.2, 0.6))] = 0
        elif r < 0.5:                                             # sensor reads through the object
            sel = gm[j] & (smooth_noise(1, 12, 16, dev)[0] > rng.uniform(0.0, 0.5))
            depth[j][sel] = far[j][sel] * (1 + 0.02 * torch.randn_like(far[j][sel]))
        if rng.random() < 0.3:                                    # small background holes
            depth[j][smooth_noise(1, 24, 32, dev)[0] > 0.92] = 0
    if rng.random() < 0.2:
        rgb = F.avg_pool2d(rgb, 3, 1, 1)
    return rgb, depth, gt, gm


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--init", required=True, help="TDCNet weights (tdcnet_repro.pth)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    dev, amp = setup(0)
    rng = random.Random(0)
    K = intrinsics_table(a.data, dev)
    net = DepthNet(True).to(dev)
    sd = torch.load(a.init, map_location="cpu", weights_only=False)["model_state_dict"]
    miss, extra = net.load_state_dict(sd, strict=False)
    assert all(k.startswith("proj") for k in miss) and not extra
    net.freeze_base()
    feats = DAFeatures().to(dev)
    dl = DataLoader(TransCG320(a.data, "train", train=True), a.bs, shuffle=True, drop_last=True, num_workers=a.workers,
                    pin_memory=True, generator=torch.Generator().manual_seed(0))
    opt = torch.optim.AdamW(net.proj.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    for ep in range(a.epochs):
        t0 = time.time(); tl = 0.0; n = 0
        for b in dl:
            b = to_device(b, dev)
            rgb, depth, gt, gm = sensor_failure_aug(b["rgb"], b["depth"], b["gt"], b["mask"], rng)
            with torch.no_grad(), torch.autocast("cuda", dtype=amp):
                f = [x.float() for x in feats(rgb)]
            loss, l1, _ = tdcnet_loss(net(rgb, depth, f), gt, gm, b["cam"], K)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.proj.parameters(), 1.0); opt.step(); sch.step()
            tl += l1; n += 1
        print(f"epoch {ep + 1}/{a.epochs} L1 {tl / n * 1000:.2f} mm {(time.time() - t0) / 60:.1f} min")
    torch.save({"model_state_dict": net.state_dict()}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
