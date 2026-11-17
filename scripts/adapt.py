"""Label-free adaptation to a new domain (ClearGrasp real images), then evaluation on ClearGrasp real-test.

No ground-truth depth or mask from the new domain is used. Only the DA injection projections (0.41M) are updated,
600 steps, batch 8, AdamW 2e-4 (wd 1e-4), grad clip 1.0, with three losses:
  * outside the predicted mask (dilated 6 px) the sensor is trusted:  L1(pred, sensor depth)   x1.0
  * inside the predicted mask: left-right flip consistency                                     x0.5
  * a TransCG batch with its usual loss, so the model keeps its TransCG accuracy               x1.0
Images: ClearGrasp real-val (173, or a random subset); evaluation: real-test (113), never seen.

    python scripts/adapt.py --cleargrasp data/cleargrasp --init checkpoints/depth_da.pth --n 173 --out checkpoints/depth_da_cg173.pth
    python scripts/adapt.py --cleargrasp data/cleargrasp --init checkpoints/adapter_aug.pth --n 173 --out checkpoints/adapter_aug_cg173.pth
"""
import argparse
import json
import random
import time

from _common import setup, add_common_args

import torch
from torch.utils.data import DataLoader

from tdc.data import ClearGraspReal, TransCG320, cleargrasp_real_files, intrinsics_table, to_device
from tdc.gate import dilate, gate, hysteresis, mask_from_prob
from tdc.losses import tdcnet_loss
from tdc.metrics import RegionMetrics, StandardMetrics
from tdc.models import DAFeatures, DepthNet, MaskHead, load_state

flip = lambda x: torch.flip(x, [-1])
CG_CAMS = {1: "D435", 2: "D415"}


@torch.no_grad()
def load_images(files, feats, head, dev, bs=16):
    out = []
    for b in DataLoader(ClearGraspReal(files), bs, shuffle=False, num_workers=4):
        b = to_device(b, dev)
        b["f"] = feats(b["rgb"])
        b["p"] = torch.sigmoid(head(b["rgb"], b["depth"], b["f"])).float()
        out.append(b)
    return out


@torch.no_grad()
def evaluate(net, batches):
    std, reg = StandardMetrics(CG_CAMS), RegionMetrics(CG_CAMS)
    for b in batches:
        m = mask_from_prob(b["p"])
        o = gate(net(b["rgb"], b["depth"], b["f"]), b["depth"], m)
        cams = b["cam"].tolist()
        std.add("m", o, b["gt"], b["mask"], cams); reg.add("m", o, b["gt"], b["mask"], cams, pred_mask=m)
    return std.summary("m"), reg.summary("m")


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--cleargrasp", required=True, help="folder with cleargrasp-dataset-test-val (real-val, real-test)")
    ap.add_argument("--init", required=True)
    ap.add_argument("--n", type=int, default=173, help="number of real-val images to adapt on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    dev, amp = setup(0)
    rng = random.Random(0)
    K = intrinsics_table(a.data, dev)
    feats = DAFeatures().to(dev)
    head = load_state(MaskHead(), f"{a.ckpt}/mask_head.pth").to(dev).eval()
    net = load_state(DepthNet(True), a.init).to(dev).eval()

    val = cleargrasp_real_files(a.cleargrasp, "real-val")
    tst = cleargrasp_real_files(a.cleargrasp, "real-test")
    vf = val if a.n >= len(val) else random.Random(a.n).sample(val, a.n)
    print(f"adapt on {len(vf)} real-val images, evaluate on {len(tst)} real-test images")
    test_b = load_images(tst, feats, head, dev)
    before = evaluate(net, test_b)

    vb = load_images(vf, feats, head, dev)
    vrgb = torch.cat([b["rgb"] for b in vb]); vdep = torch.cat([b["depth"] for b in vb]); vp = torch.cat([b["p"] for b in vb])
    anchor_dl = DataLoader(TransCG320(a.data, "train", train=True, color=False), a.bs, shuffle=True, drop_last=True,
                           num_workers=4, generator=torch.Generator().manual_seed(1))
    anchor = iter(anchor_dl)
    net.freeze_base()
    opt = torch.optim.AdamW(net.proj.parameters(), lr=a.lr, weight_decay=1e-4)
    t0 = time.time()
    for it in range(a.steps):
        j = torch.randint(0, len(vrgb), (a.bs,), device=dev)
        rgb, depth, p = vrgb[j], vdep[j], vp[j]
        if rng.random() < 0.5:
            rgb, depth, p = flip(rgb), flip(depth), flip(p)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            f0 = [x.float() for x in feats(rgb)]; f1 = [x.float() for x in feats(flip(rgb))]
        pred = net(rgb, depth, f0)
        with torch.no_grad():
            pf = flip(net(flip(rgb), flip(depth), f1))
        m = hysteresis(p)
        outside = ~dilate(m, 6) & (depth > 0.3) & (depth < 1.5)
        l_out = (pred - depth).abs()[outside].mean() if outside.any() else pred.sum() * 0
        l_cons = (pred - (pred.detach() + pf) / 2).abs()[dilate(m, 3)].mean() if m.any() else pred.sum() * 0
        try:
            ab = next(anchor)
        except StopIteration:
            anchor = iter(anchor_dl); ab = next(anchor)
        ab = to_device(ab, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            fa = [x.float() for x in feats(ab["rgb"])]
        l_anc = tdcnet_loss(net(ab["rgb"], ab["depth"], fa), ab["gt"], ab["mask"], ab["cam"], K)[0]
        loss = 1.0 * l_out + 0.5 * l_cons + 1.0 * l_anc
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.proj.parameters(), 1.0); opt.step()
        if (it + 1) % 100 == 0:
            print(f"step {it + 1}/{a.steps} outside {float(l_out) * 1000:.2f} consistency {float(l_cons) * 1000:.2f} "
                  f"TransCG {float(l_anc) * 1000:.2f} mm | {(time.time() - t0) / 60:.1f} min")
    net.eval()
    torch.save({"model_state_dict": net.state_dict()}, a.out)
    after = evaluate(net, test_b)
    print(f"\nClearGrasp real-test transparent RMSE: {before[0]['rmse']:.2f} -> {after[0]['rmse']:.2f} mm "
          f"(background {before[1]['background']:.2f} -> {after[1]['background']:.2f} mm)")
    json.dump({"before": before, "after": after, "n_adapt": len(vf)}, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
