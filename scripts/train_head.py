"""Train the transparency mask head on frozen Depth Anything V2-Base features.

    python scripts/train_head.py --out checkpoints/mask_head.pth                                    # 252x336 head
    python scripts/train_head.py --hi --init checkpoints/mask_head.pth --out checkpoints/mask_head_hi.pth   # 378x504 head

Base head : every 2nd training image, 10 epochs, batch 16, AdamW 1e-3, cosine.
Hi-res    : DA at 378x504, initialised from the base head, every 4th image, 4 epochs, AdamW 3e-4, cosine.
Loss: BCE (pos_weight 10) + Dice against the GT transparent mask. Flip augmentation only (no colour jitter).
"""
import argparse
import math
import time

from _common import setup, add_common_args

import torch
from torch.utils.data import DataLoader

from tdc import DA_SIZE, DA_SIZE_HI
from tdc.data import TransCG320, to_device
from tdc.losses import mask_loss
from tdc.models import DAFeatures, MaskHead, load_state


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--hi", action="store_true", help="378x504 Depth Anything input")
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--every", type=int, default=None, help="use every k-th training image")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    epochs = a.epochs or (4 if a.hi else 10)
    lr = a.lr or (3e-4 if a.hi else 1e-3)
    every = a.every or (4 if a.hi else 2)
    dev, amp = setup(0)

    feats = DAFeatures(size=DA_SIZE_HI if a.hi else DA_SIZE).to(dev)
    head = MaskHead(c_da=feats.channels).to(dev)
    if a.init:
        load_state(head, a.init)
    ds = TransCG320(a.data, "train", train=True, color=False)
    ds.items = ds.items[::every]
    dl = DataLoader(ds, a.bs, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    steps = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    print(f"train {len(ds)} images, {epochs} epochs, lr {lr}, DA {feats.size}")
    for ep in range(epochs):
        head.train(); t0 = time.time(); tl = 0.0; n = 0
        for b in dl:
            b = to_device(b, dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=amp):
                f = [x.float() for x in feats(b["rgb"])]
            loss = mask_loss(head(b["rgb"], b["depth"], f), b["mask"].float())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item(); n += 1
        print(f"epoch {ep + 1}/{epochs} loss {tl / n:.4f} {(time.time() - t0) / 60:.1f} min")
    torch.save({"model_state_dict": head.state_dict()}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
