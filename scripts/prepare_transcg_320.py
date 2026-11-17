"""Resize the original TransCG dataset (1280x720) to 320x240, keeping its folder layout and PNG depth units.

    python scripts/prepare_transcg_320.py --src /path/to/TransCG --dst data/transcg_320

All images are resized with nearest-neighbour interpolation, as the official TransCG loader does when it resizes to
(320, 240). Note: the cache used for the reported numbers was built separately; this script recreates the same
layout and resolution, but tiny numeric differences are possible.
"""
import argparse
import glob
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2

FILES = ["rgb{c}.png", "depth{c}.png", "depth{c}-gt.png", "depth{c}-gt-mask.png"]


def convert(args):
    src_dir, dst_dir = args
    os.makedirs(dst_dir, exist_ok=True)
    for c in (1, 2):
        for f in FILES:
            s = os.path.join(src_dir, f.format(c=c))
            if not os.path.exists(s):
                continue
            flag = cv2.IMREAD_COLOR if f.startswith("rgb") else cv2.IMREAD_UNCHANGED
            x = cv2.imread(s, flag)
            cv2.imwrite(os.path.join(dst_dir, f.format(c=c)), cv2.resize(x, (320, 240), interpolation=cv2.INTER_NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/transcg_320")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    shutil.copytree(f"{a.src}/camera_intrinsics", f"{a.dst}/camera_intrinsics", dirs_exist_ok=True)
    if os.path.exists(f"{a.src}/metadata.json"):
        shutil.copy(f"{a.src}/metadata.json", a.dst)
    jobs = []
    for scene in sorted(glob.glob(f"{a.src}/scene*")):
        name = os.path.basename(scene)
        os.makedirs(f"{a.dst}/{name}", exist_ok=True)
        shutil.copy(f"{scene}/metadata.json", f"{a.dst}/{name}/")
        meta = json.load(open(f"{scene}/metadata.json"))
        persp = set(meta["D435_valid_perspective_list"]) | set(meta["L515_valid_perspective_list"])
        jobs += [(f"{scene}/{p}", f"{a.dst}/{name}/{p}") for p in sorted(persp)]
    with ThreadPoolExecutor(a.threads) as ex:
        for i, _ in enumerate(ex.map(convert, jobs)):
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)}")
    print("done:", a.dst)


if __name__ == "__main__":
    main()
