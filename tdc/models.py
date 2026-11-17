"""Networks: Depth Anything V2 feature extractor, TDCNet (+ DA feature injection), mask head, distilled student."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import H, W, DA_SIZE
from .third_party import TDCNet, in_remake_dir

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize(rgb, size):
    mean = rgb.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = rgb.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (F.interpolate(rgb, size=size, mode="bilinear", align_corners=False) - mean) / std


class DAFeatures(nn.Module):
    """Frozen Depth Anything V2 encoder. Returns the 4 intermediate feature maps the HF backbone exposes
    (layers 3, 6, 9, 12 for ViT-B), each as (N, C, size/14, size/14). Only the encoder runs: the DPT decoder
    that produces relative depth is skipped, which is where most of the speed comes from."""

    def __init__(self, name="depth-anything/Depth-Anything-V2-Base-hf", size=DA_SIZE, model=None):
        super().__init__()
        if model is None:
            from transformers import AutoModelForDepthEstimation
            model = AutoModelForDepthEstimation.from_pretrained(name)
        self.backbone = model.backbone          # share the backbone between several extractors (different sizes)
        self.size = tuple(size)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):                 # always frozen
        return super().train(False)

    @property
    def channels(self):
        return self.backbone.config.hidden_size

    def forward(self, rgb):
        hp, wp = self.size[0] // 14, self.size[1] // 14
        fms = self.backbone(_normalize(rgb, self.size)).feature_maps
        out = []
        for fm in fms:
            if fm.dim() == 3:                   # (N, tokens, C) -> (N, C, hp, wp), drop the CLS token
                fm = fm[:, -hp * wp:].reshape(fm.shape[0], hp, wp, -1).permute(0, 3, 1, 2)
            out.append(fm.float())
        return out


class DepthNet(nn.Module):
    """TDCNet, optionally with Depth Anything feature injection.

    TDCNet sums its Transformer-branch (RGB-D) and CNN-branch (depth) features at 4 scales before the decoder.
    With ``use_da=True`` we add a projection of DA feature i (3/6/9/12th layer) to scale i:
    1x1 conv C_da->64, GroupNorm, GELU, 3x3 conv 64->c_i, bilinearly upsampled. The last conv is zero-initialised,
    so training starts exactly from plain TDCNet. 0.41M extra parameters.
    """

    def __init__(self, use_da=True, c_da=768, c_mid=64):
        super().__init__()
        self.use_da = use_da
        with in_remake_dir():                   # TDCNet is built as in the original experiments (cwd = ReMake repo)
            self.base = TDCNet()(lambda_val=1, res=True)
        if use_da:
            self.base.eval()
            with torch.no_grad():
                chs = [e.shape[1] for e in self.base.encoder(torch.zeros(1, 4, H, W))]
            self.base.train()
            self.proj = nn.ModuleList()
            for c in chs:
                m = nn.Sequential(nn.Conv2d(c_da, c_mid, 1), nn.GroupNorm(8, c_mid), nn.GELU(), nn.Conv2d(c_mid, c, 3, padding=1))
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)
                self.proj.append(m)

    def forward(self, rgb, depth, feats=None):
        """rgb (N,3,H,W) in [0,1], depth (N,H,W) in metres, feats: list of 4 DA maps. Returns (N,H,W) metres."""
        if not self.use_da:
            return self.base(rgb, depth).float().view_as(depth)
        n, h, w = depth.shape
        d = depth.view(n, 1, h, w)
        enc = self.base.encoder(torch.cat((rgb, d), 1))
        res = self.base.resnet(d)
        for i in range(4):
            enc[i] = enc[i] + res[i] + F.interpolate(self.proj[i](feats[i]), size=enc[i].shape[-2:], mode="bilinear", align_corners=False)
        return self.base.final(self.base.decoder(enc)).float().view_as(depth)

    def freeze_base(self):
        """Train only the injection projections (used by the strong-augmentation adapter and test-time adaptation)."""
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith("proj")
        self.train()
        self.base.eval()
        return self


class MaskHead(nn.Module):
    """Predicts the transparent-object mask without ground truth.
    Each DA feature map is reduced to 32 channels and upsampled to 320x240, concatenated with RGB, raw depth and a
    depth-valid flag, then 4 dilated 3x3 conv blocks (dilation 1, 2, 4, 1). Output: logits (N,H,W)."""

    def __init__(self, c_da=768, c_red=32):
        super().__init__()
        self.red = nn.ModuleList([nn.Conv2d(c_da, c_red, 1) for _ in range(4)])
        blk = lambda i, o, dl=1: [nn.Conv2d(i, o, 3, padding=dl, dilation=dl), nn.GroupNorm(8, o), nn.ReLU(True)]
        self.net = nn.Sequential(*blk(4 * c_red + 5, 64), *blk(64, 64, 2), *blk(64, 64, 4), *blk(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, rgb, depth, feats):
        h, w = depth.shape[-2:]
        d1 = depth[:, None]
        valid = (d1 > 0).float()
        f = [F.interpolate(r(x), size=(h, w), mode="bilinear", align_corners=False) for r, x in zip(self.red, feats)]
        return self.net(torch.cat(f + [rgb, d1, valid], 1))[:, 0]


class Student(nn.Module):
    """ResNet-18 that imitates the 4 Depth Anything V2-Base feature maps (768ch, 18x24) for distillation.
    Layers 2-4 are resized to the DA grid and mapped by small adapters to 768 channels."""

    def __init__(self, pretrained=True, size=DA_SIZE):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.size = tuple(size)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        ad = lambda i: nn.Sequential(nn.Conv2d(i, 256, 3, 1, 1), nn.GroupNorm(16, 256), nn.ReLU(True), nn.Conv2d(256, 768, 1))
        self.ad = nn.ModuleList([ad(128 + 256), ad(256 + 512), ad(256 + 512), ad(512)])

    def forward(self, rgb):
        hp, wp = self.size[0] // 14, self.size[1] // 14
        x = self.stem(_normalize(rgb, self.size))
        a = self.l1(x); b = self.l2(a); c = self.l3(b); d = self.l4(c)
        rs = lambda t: F.interpolate(t, size=(hp, wp), mode="bilinear", align_corners=False)
        ins = [torch.cat([rs(b), rs(c)], 1), torch.cat([rs(c), rs(d)], 1), torch.cat([rs(c), rs(d)], 1), rs(d)]
        return [m(t).float() for m, t in zip(self.ad, ins)]


def load_state(module, path, key="model_state_dict", strict=True, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    sd = ck[key] if isinstance(ck, dict) and key in ck else ck
    module.load_state_dict(sd, strict=strict)
    return module
