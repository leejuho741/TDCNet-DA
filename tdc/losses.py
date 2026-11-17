"""Training losses."""
import torch
import torch.nn.functional as F

from . import H, W
from .third_party import surface_normal_fn

_NORMAL = None
_GRID = {}


def _grid(device):
    if device not in _GRID:
        ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
        _GRID[device] = (ys, xs)
    return _GRID[device]


def normals(depth, cams, K):
    """Surface normals from depth with per-camera intrinsics K (see data.intrinsics_table), scaled 1280x720 -> 320x240."""
    global _NORMAL
    if _NORMAL is None:
        _NORMAL = surface_normal_fn()
    ys, xs = _grid(depth.device)
    k = K[cams]
    sx, sy = W / 1280.0, H / 720.0
    fx, fy = (k[:, 0] * sx)[:, None, None], (k[:, 1] * sy)[:, None, None]
    cx, cy = (k[:, 2] * sx)[:, None, None], (k[:, 3] * sy)[:, None, None]
    xyz = torch.stack([(xs[None] - cx) * depth / fx, (ys[None] - cy) * depth / fy, depth], 1)
    return _NORMAL(xyz)


def tdcnet_loss(pred, gt, gt_mask, cams, K, beta=0.1):
    """TDCNet's official loss: L1 inside the transparent mask (GT > 0) + 0.1 x (1 - cos) of surface normals.
    Pixels outside the mask are not supervised."""
    zm = gt >= 0.01
    lm = gt_mask & zm
    zmd = ~(F.max_pool2d((~zm)[:, None].float(), 3, 1, 1)[:, 0] > 0.5)
    lmd = gt_mask & zmd
    l1 = (pred - gt).abs()[lm].mean() if lm.any() else pred.sum() * 0
    with torch.no_grad():
        ngt = normals(gt, cams, K)
    sn = (1 - F.cosine_similarity(normals(pred, cams, K), ngt, dim=1))[lmd].mean() if lmd.any() else pred.sum() * 0
    return l1 + beta * sn, float(l1), float(sn)


def dice_loss(logit, target):
    p = torch.sigmoid(logit)
    return (1 - (2 * (p * target).sum((1, 2)) + 1) / (p.sum((1, 2)) + target.sum((1, 2)) + 1)).mean()


def mask_loss(logit, target, pos_weight=10.0):
    """Mask head loss: BCE with pos_weight 10 + Dice."""
    pw = torch.tensor(pos_weight, device=logit.device)
    return F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw) + dice_loss(logit, target)


def feature_loss(student, teacher):
    """Distillation: per-layer MSE normalised by the teacher's feature energy, averaged over the 4 layers."""
    return sum(((s - t) ** 2).mean() / (t ** 2).mean().clamp(min=1e-6) for s, t in zip(student, teacher)) / len(teacher)
