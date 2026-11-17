"""Distil Depth Anything V2-Base (86.6M, teacher) into a ResNet-18 feature extractor (17.6M, student).

Stage 1 (student + a copy of the mask head; the depth model stays frozen):
  TransCG batch : feature imitation (energy-normalised MSE, 4 layers)                 x1.0
                  depth-output imitation through the frozen depth model (L1, 4x inside the mask) x1.0
                  GT depth loss through the frozen depth model                          x0.5
                  mask: BCE to the teacher head + BCE(pos_weight 10) to GT + Dice      x0.5
  COCO batch    : feature imitation only (RGB, no labels needed)                        x1.0
  6 epochs over TransCG train, batch 8, AdamW (student 2e-4, head 5e-5), OneCycle, grad clip 1.0.
Stage 2 (student frozen): re-tune the depth model's injection projections + head on half of TransCG for 1 epoch.

The reported student was trained in two rounds (6 epochs on half of TransCG, then this recipe continued from it).
Pass the first round's weights with --init to reproduce that; from ImageNet weights expect slightly different numbers.

    python scripts/distill.py --coco data/coco/val2017 --out runs/student
"""
import argparse
import copy
import glob
import os
import time

from _common import setup, add_common_args

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tdc import H, W
from tdc.data import TransCG320, intrinsics_table, to_device
from tdc.losses import feature_loss, tdcnet_loss
from tdc.models import DAFeatures, DepthNet, MaskHead, Student, load_state


class RGBImages(torch.utils.data.Dataset):
    """Unlabelled photos: random 4:3 crop (50-100%), flip, colour jitter, resized to 320x240."""

    def __init__(self, files):
        self.f = files

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        im = cv2.imread(self.f[i])
        if im is None:
            im = np.zeros((480, 640, 3), np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB); h, w = im.shape[:2]; r = np.random
        s = r.uniform(0.5, 1.0); ch = int(min(h, w * 3 / 4) * s); cw = int(ch * 4 / 3)
        y = r.randint(0, h - ch + 1); x = r.randint(0, w - cw + 1)
        im = cv2.resize(im[y:y + ch, x:x + cw], (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if r.rand() < 0.5:
            im = im[:, ::-1]
        if r.rand() < 0.8:
            im = np.clip((im - 0.5) * r.uniform(0.8, 1.2) + 0.5 + r.uniform(-0.1, 0.1), 0, 1) * r.uniform(0.9, 1.1, 3)
        return torch.from_numpy(np.ascontiguousarray(np.clip(im, 0, 1), dtype=np.float32)).permute(2, 0, 1)


def depth_imitation(D, tD, gm):
    return (D - tD).abs().mean() + 4 * ((D - tD).abs()[gm].mean() if gm.any() else D.sum() * 0)


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--coco", default=None, help="folder of unlabelled photos (COCO val2017). Omit for TransCG only")
    ap.add_argument("--init", default=None, help="student weights to continue from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    dev, amp = setup(0)
    os.makedirs(a.out, exist_ok=True)
    K = intrinsics_table(a.data, dev)
    LR_S, LR_H, LR_P, LR_H2 = 2e-4, 5e-5, 1e-4, 1e-4

    teacher = DAFeatures().to(dev)
    depth_net = load_state(DepthNet(True), f"{a.ckpt}/depth_da.pth").to(dev).eval()
    head_t = load_state(MaskHead(), f"{a.ckpt}/mask_head.pth").to(dev).eval()
    for m in (depth_net, head_t):
        for p in m.parameters():
            p.requires_grad = False
    stu = Student(pretrained=a.init is None).to(dev)
    if a.init:
        load_state(stu, a.init)
    head_s = copy.deepcopy(head_t)
    for p in head_s.parameters():
        p.requires_grad = True

    ext = None
    if a.coco:
        files = sorted(glob.glob(f"{a.coco}/*.jpg"))
        print("unlabelled photos:", len(files))

        def ext_iter():
            while True:
                for x in DataLoader(RGBImages(files), a.bs, shuffle=True, drop_last=True, num_workers=6, pin_memory=True):
                    yield x
        ext = ext_iter()

    def teach(rgb):
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            return [x.float() for x in teacher(rgb)]

    pos_w = torch.tensor(10.0, device=dev)
    # ---- stage 1 ----
    ck1 = f"{a.out}/stage1.pth"
    if os.path.exists(ck1):
        c = torch.load(ck1, map_location=dev, weights_only=False); stu.load_state_dict(c["student"]); head_s.load_state_dict(c["head"])
    else:
        dl = DataLoader(TransCG320(a.data, "train", train=True), a.bs, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True)
        opt = torch.optim.AdamW([{"params": stu.parameters(), "lr": LR_S}, {"params": head_s.parameters(), "lr": LR_H}], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_S, LR_H], total_steps=a.epochs * len(dl), pct_start=0.03)
        for ep in range(a.epochs):
            stu.train(); head_s.train(); t0 = time.time(); S = np.zeros(4); n = 0
            for b in dl:
                b = to_device(b, dev); rgb, depth, gt, gm, cams = b["rgb"], b["depth"], b["gt"], b["mask"], b["cam"]
                tf = teach(rgb)
                with torch.no_grad():
                    tp = torch.sigmoid(head_t(rgb, depth, tf)).float(); tD = depth_net(rgb, depth, tf)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sf = stu(rgb)
                sf = [x.float() for x in sf]
                D = depth_net(rgb, depth, sf)
                logit = head_s(rgb, depth, sf); p = torch.sigmoid(logit); gmf = gm.float()
                dice = 1 - (2 * (p * gmf).sum((1, 2)) + 1) / (p.sum((1, 2)) + gmf.sum((1, 2)) + 1)
                l_mask = F.binary_cross_entropy_with_logits(logit, tp) + F.binary_cross_entropy_with_logits(logit, gmf, pos_weight=pos_w) + dice.mean()
                l_feat = feature_loss(sf, tf)
                loss = l_feat + depth_imitation(D, tD, gm) + 0.5 * tdcnet_loss(D, gt, gm, cams, K)[0] + 0.5 * l_mask
                if ext is not None:
                    xe = next(ext).to(dev, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        se = stu(xe)
                    loss = loss + feature_loss([x.float() for x in se], teach(xe))
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(stu.parameters()) + list(head_s.parameters()), 1.0); opt.step(); sched.step()
                S += [float(l_feat), float(l_mask), 0, 0]; n += 1
            print(f"stage 1 epoch {ep + 1}/{a.epochs} feature {S[0] / n:.4f} mask {S[1] / n:.3f} | {(time.time() - t0) / 60:.1f} min")
        torch.save({"student": stu.state_dict(), "head": head_s.state_dict()}, ck1)
    stu.eval()
    for p in stu.parameters():
        p.requires_grad = False

    # ---- stage 2 ----
    dS = copy.deepcopy(depth_net); dS.freeze_base(); hS = copy.deepcopy(head_s)
    for p in hS.parameters():
        p.requires_grad = True
    ds2 = TransCG320(a.data, "train", train=True); ds2.items = ds2.items[::2]
    dl2 = DataLoader(ds2, a.bs, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True)
    opt2 = torch.optim.AdamW([{"params": dS.proj.parameters(), "lr": LR_P}, {"params": hS.parameters(), "lr": LR_H2}], weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.OneCycleLR(opt2, max_lr=[LR_P, LR_H2], total_steps=len(dl2), pct_start=0.05)
    for b in dl2:
        b = to_device(b, dev); rgb, depth, gt, gm, cams = b["rgb"], b["depth"], b["gt"], b["mask"], b["cam"]
        tf = teach(rgb)
        with torch.no_grad():
            tp = torch.sigmoid(head_t(rgb, depth, tf)).float(); tD = depth_net(rgb, depth, tf); sf = stu(rgb)
        D = dS(rgb, depth, sf)
        logit = hS(rgb, depth, sf)
        l_mask = F.binary_cross_entropy_with_logits(logit, tp) + F.binary_cross_entropy_with_logits(logit, gm.float(), pos_weight=pos_w)
        loss = depth_imitation(D, tD, gm) + 0.5 * tdcnet_loss(D, gt, gm, cams, K)[0] + 0.5 * l_mask
        opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step(); sch2.step()
    for name, mod in [("student", stu), ("student_depth", dS), ("student_head", hS)]:
        torch.save({"model_state_dict": mod.state_dict()}, f"{a.out}/{name}.pth")
    print("saved student.pth / student_depth.pth / student_head.pth in", a.out)


if __name__ == "__main__":
    main()
