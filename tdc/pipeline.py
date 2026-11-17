"""End-to-end model: RGB-D in, completed depth out, no ground-truth mask needed.

fast     : one Depth Anything pass (252x336) shared by the depth model and the mask head.      ~66 FPS on an L4
accurate : the mask head gets its own Depth Anything pass at 378x504, plus left-right flip TTA. ~31 FPS on an L4
student  : a distilled ResNet-18 replaces Depth Anything (no DA at inference).                  ~85 FPS on an L4
"""
import os

import torch
import torch.nn as nn

from . import DA_SIZE, DA_SIZE_HI
from .gate import gate, mask_from_prob, HYST, ERODE, BLEND
from .models import DAFeatures, DepthNet, MaskHead, Student, load_state

CKPT_FILES = {
    "tdcnet": "tdcnet_repro.pth",          # TDCNet reproduced with the official recipe (baseline)
    "depth": "depth_da.pth",               # TDCNet + DA injection, trained from scratch (final depth model)
    "head": "mask_head.pth",               # mask head on DA features at 252x336
    "head_hi": "mask_head_hi.pth",         # mask head on DA features at 378x504
    "adapter_aug": "adapter_aug.pth",      # frozen TDCNet + injection trained with strong augmentation
    "student": "student.pth",              # distilled ResNet-18 feature extractor
    "student_depth": "student_depth.pth",  # depth model with projections re-tuned to the student
    "student_head": "student_head.pth",    # mask head re-tuned to the student
}

flip = lambda x: torch.flip(x, [-1])


class Completer(nn.Module):
    def __init__(self, depth_net, head, feats, head_feats=None, tta=False, hyst=HYST, erode_px=ERODE, blend=BLEND):
        super().__init__()
        self.depth_net, self.head, self.feats = depth_net, head, feats
        self.head_feats = head_feats          # None: the head reuses the depth model's features
        self.tta, self.hyst, self.erode_px, self.blend = tta, hyst, erode_px, blend

    @torch.no_grad()
    def forward(self, rgb, depth, return_all=False):
        f = self.feats(rgb)
        d = self.depth_net(rgb, depth, f)
        if self.tta:
            d = (d + flip(self.depth_net(flip(rgb), flip(depth), self.feats(flip(rgb))))) / 2
        hf = f if self.head_feats is None else self.head_feats(rgb)
        prob = torch.sigmoid(self.head(rgb, depth, hf)).float()
        m = mask_from_prob(prob, self.hyst, self.erode_px)
        out = gate(d, depth, m, self.blend)
        return (out, dict(depth_net=d, prob=prob, mask=m)) if return_all else out


def build(mode="fast", ckpt_dir="checkpoints", device="cuda", da_name="depth-anything/Depth-Anything-V2-Base-hf"):
    p = lambda k: os.path.join(ckpt_dir, CKPT_FILES[k])
    if mode == "student":
        feats = load_state(Student(pretrained=False), p("student")).to(device).eval()
        depth_net = load_state(DepthNet(True), p("student_depth")).to(device).eval()
        head = load_state(MaskHead(), p("student_head")).to(device).eval()
        return Completer(depth_net, head, feats).eval()
    feats = DAFeatures(da_name, DA_SIZE).to(device)
    depth_net = load_state(DepthNet(True), p("depth")).to(device).eval()
    if mode == "fast":
        head = load_state(MaskHead(), p("head")).to(device).eval()
        return Completer(depth_net, head, feats).eval()
    if mode == "accurate":
        head = load_state(MaskHead(), p("head_hi")).to(device).eval()
        head_feats = DAFeatures(size=DA_SIZE_HI, model=_wrap(feats)).to(device)
        return Completer(depth_net, head, feats, head_feats=head_feats, tta=True).eval()
    raise ValueError(mode)


class _wrap:                       # lets a second DAFeatures share the first one's backbone (no second copy in memory)
    def __init__(self, f):
        self.backbone = f.backbone
