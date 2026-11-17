"""Train the depth model with TDCNet's official recipe.

    python scripts/train_depth.py --da --seed 0 --out runs/depth_da          # final model (TDCNet + DA injection)
    python scripts/train_depth.py --no-da --out runs/tdcnet_repro            # TDCNet baseline reproduction

Recipe (identical for both): official TransCG loader and augmentation at 320x240, AdamW lr 1e-3, batch 8,
40 epochs, lr x0.1 at epoch 35, fp32 (TF32), loss = L1 + 0.1 x surface-normal term inside the transparent mask.
With --da, frozen Depth Anything V2-Base features are injected at the 4 encoder scales (zero-initialised).
About 13-15 h on one L4. Resumes from <out>/last.pth.
"""
import argparse
import os
import time

from _common import setup, add_common_args

import torch
from torch.utils.data import DataLoader

from tdc import H, W
from tdc.data import intrinsics_table, to_device
from tdc.losses import tdcnet_loss
from tdc.models import DAFeatures, DepthNet
from tdc.third_party import official_transcg, patch_numpy2


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--da", dest="da", action="store_true", default=True)
    ap.add_argument("--no-da", dest="da", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--milestone", type=int, default=35)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev, amp = setup(a.seed)
    os.makedirs(a.out, exist_ok=True)
    K = intrinsics_table(a.data, dev)

    net = DepthNet(use_da=a.da).to(dev)
    feats = DAFeatures().to(dev) if a.da else None
    print(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M"
          + (f" (injection {sum(p.numel() for n, p in net.named_parameters() if n.startswith('proj')) / 1e6:.2f}M)" if a.da else ""))

    print("numpy-2 patch on the official augmentation code:", patch_numpy2(), "place(s)")
    ds = official_transcg(a.data, split="train", image_size=(320, 240), use_augmentation=True, rgb_augmentation_probability=0.8,
                          use_depth_augmentation=False, depth_min=0.0, depth_max=10.0, depth_norm=1.0, reldepth_model=None)
    dl = DataLoader(ds, a.bs, shuffle=True, num_workers=a.workers, pin_memory=True, persistent_workers=True,
                    generator=torch.Generator().manual_seed(a.seed))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[a.milestone], gamma=0.1)
    start, last = 0, f"{a.out}/last.pth"
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); start = ck["epoch"]
        print(f"resume from epoch {start}")
    print(f"train {len(ds)} images, {len(dl)} steps/epoch, {a.epochs} epochs")

    for ep in range(start, a.epochs):
        net.train(); t0 = time.time(); s1 = s2 = 0.0; n = 0
        for b in dl:
            b = to_device(b, dev)
            rgb = b["rgb"]
            depth = b["depth"].view(rgb.shape[0], H, W)
            gt = b["depth_gt"].view_as(depth)
            gt_m = (b["depth_gt_mask"] > 0.5).view_as(depth)
            cams = torch.where((b["fx"].float() - K[1, 0]).abs() < 1.0, 1, 2).long()    # loader gives fx; D435 vs L515
            f = None
            if a.da:
                with torch.no_grad(), torch.autocast("cuda", dtype=amp):
                    f = [x.float() for x in feats(rgb)]
            loss, l1, sn = tdcnet_loss(net(rgb, depth, f), gt, gt_m, cams, K)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            s1 += l1; s2 += sn; n += 1
        sched.step()
        print(f"epoch {ep + 1}/{a.epochs} {(time.time() - t0) / 60:.1f} min | L1 {s1 / n * 1000:.1f} mm normal {s2 / n:.3f}")
        torch.save({"model_state_dict": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1}, last)
    torch.save({"model_state_dict": net.state_dict()}, f"{a.out}/final.pth")
    print("saved", f"{a.out}/final.pth")


if __name__ == "__main__":
    main()
