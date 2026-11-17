"""Real-time transparent-object depth completion without a ground-truth mask.

TDCNet + frozen Depth Anything V2 features + a learned transparency mask head and a
distance-blended gate that keeps the sensor depth outside transparent regions.
"""
__version__ = "1.0.0"

H, W = 240, 320              # working resolution (TransCG resized to 320x240)
DA_SIZE = (252, 336)         # Depth Anything input for the depth model / fast mask head (18x24 patches)
DA_SIZE_HI = (378, 504)      # Depth Anything input for the high-resolution mask head (27x36 patches)
