"""Quick self-check after setup (no dataset needed): builds every network, runs random inputs, checks shapes,
the gate, and - if checkpoints/ is filled - that every checkpoint loads into its network.

    python scripts/smoke_test.py
"""
import os

from _common import setup

import torch

from tdc import H, W
from tdc.gate import blend_weight, gate, mask_from_prob
from tdc.models import DAFeatures, DepthNet, MaskHead, Student
from tdc.pipeline import CKPT_FILES, Completer


def main():
    dev, _ = setup(0)
    rgb, depth = torch.rand(2, 3, H, W, device=dev), torch.rand(2, H, W, device=dev) + 0.3
    feats = DAFeatures().to(dev)
    f = feats(rgb)
    assert len(f) == 4 and f[0].shape == (2, 768, 18, 24), [x.shape for x in f]
    net, head, stu = DepthNet(True).to(dev).eval(), MaskHead().to(dev).eval(), Student(pretrained=False).to(dev).eval()
    plain = DepthNet(False).to(dev).eval()
    with torch.no_grad():
        assert plain(rgb, depth).shape == (2, H, W)
        d = net(rgb, depth, f)
        # zero-initialised injection: an untrained DepthNet(True) equals plain TDCNet with the same base weights
        plain.load_state_dict({k: v for k, v in net.state_dict().items() if k.startswith("base.")})
        diff = float((d - plain(rgb, depth)).abs().max())
        print(f"zero-init injection vs plain TDCNet: max diff {diff:.2e} m (expected ~0)")
        assert head(rgb, depth, f).shape == (2, H, W)
        assert [x.shape for x in stu(rgb)] == [x.shape for x in f]
        out = Completer(net, head, feats)(rgb, depth)
        assert out.shape == (2, H, W)
    m = torch.zeros(1, H, W, dtype=torch.bool, device=dev); m[0, 100:140, 100:160] = True
    w = blend_weight(m, torch.ones(1, H, W, device=dev))
    assert w[0, 120, 130] == 1 and w[0, 120, 160 + 3] == 1 and w[0, 120, 160 + 20] == 0      # 1 up to 4 px outside, 0 past 12 px
    assert mask_from_prob(m.float()).sum() < m.sum()                                           # 2 px erosion
    assert torch.equal(gate(torch.zeros(1, H, W, device=dev), torch.ones(1, H, W, device=dev), m)[0, 0, 0], torch.tensor(1.0, device=dev))
    print("networks and gate: OK")
    ck = "checkpoints"
    if os.path.isdir(ck):
        nets = {"tdcnet": DepthNet(False), "depth": DepthNet(True), "head": MaskHead(), "head_hi": MaskHead(),
                "adapter_aug": DepthNet(True), "student": Student(pretrained=False), "student_depth": DepthNet(True), "student_head": MaskHead()}
        for k, mod in nets.items():
            p = os.path.join(ck, CKPT_FILES[k])
            if os.path.exists(p):
                mod.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"])
                print(f"  {CKPT_FILES[k]:22s} loads")
            else:
                print(f"  {CKPT_FILES[k]:22s} (not found)")


if __name__ == "__main__":
    main()
