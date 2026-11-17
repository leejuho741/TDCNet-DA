"""Mask post-processing and the gate that merges network output with the sensor depth.

Final setting (chosen by a 1,120-combination sweep, see scripts/sweep_mask.py):
hysteresis (high 0.5, low 0.1) -> erode 2 px -> blend weight 1 inside the mask, falling linearly to 0
between 4 and 12 px outside it. Pixels with no sensor depth always take the network output.
"""
import torch
import torch.nn.functional as F

HYST = (0.5, 0.1)
ERODE = 2
BLEND = (4, 12)


def dilate(m, r):
    return m if r == 0 else F.max_pool2d(m[:, None].float(), 2 * r + 1, 1, r)[:, 0] > 0.5


def erode(m, r):
    return m if r == 0 else ~dilate(~m, r)


def hysteresis(prob, hi=HYST[0], lo=HYST[1], iters=48):
    """GPU hysteresis thresholding: keep weak (>lo) regions connected to a strong (>hi) seed.
    Grows the seed by 3x3 max-pooling inside the weak region (48 steps reach 48 px)."""
    weak = (prob > lo).float()[:, None]
    m = (prob > hi).float()[:, None]
    for _ in range(iters):
        m = F.max_pool2d(m, 3, 1, 1) * weak
    return m[:, 0] > 0.5


_DISKS = {}


def _disk(r, device):
    key = (r, str(device))
    if key not in _DISKS:
        a = torch.arange(-r, r + 1, device=device)
        _DISKS[key] = ((a[:, None] ** 2 + a[None] ** 2) <= r * r).float()[None, None]
    return _DISKS[key]


def distance_to_mask(m, rmax=12):
    """Integer distance (ceil of Euclidean) from each pixel to the mask, 0 inside, rmax+1 beyond rmax."""
    d = torch.full(m.shape, float(rmax + 1), device=m.device)
    mf = m[:, None].float()
    for r in range(rmax, 0, -1):
        d = torch.where(F.conv2d(mf, _disk(r, m.device), padding=r)[:, 0] > 0.5, torch.full_like(d, float(r)), d)
    return torch.where(m, torch.zeros_like(d), d)


def blend_weight(m, depth, blend=BLEND):
    a, b = blend
    dd = distance_to_mask(m, rmax=b)
    w = ((b - dd) / max(b - a, 1e-6)).clamp(0, 1)
    return torch.where(depth > 0, w, torch.ones_like(w))


def gate(out, depth, m, blend=BLEND):
    """Inside the (blended) mask use the network, outside keep the sensor value."""
    w = blend_weight(m, depth, blend)
    return w * out + (1 - w) * depth


def mask_from_prob(prob, hyst=HYST, erode_px=ERODE):
    return erode(hysteresis(prob, *hyst), erode_px)
