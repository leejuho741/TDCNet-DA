"""Copy the trained weights from the experiment folder layout (Google Drive, tdc_finetune/) into checkpoints/
with stable names, keeping only the model weights (no optimizer state).

    python scripts/export_checkpoints.py --src /content/drive/MyDrive/tdc_finetune --dst checkpoints
"""
import argparse
import os

import torch

# clean name -> (relative path in the experiment folder, key inside the file)
MAP = {
    "tdcnet_repro.pth": ("b2_tdcnet/repro_fp32/final.tar", "model_state_dict"),
    "depth_da.pth": ("next12_scratch/s0/final.tar", "model_state_dict"),
    "mask_head.pth": ("b3_head/head_dav2-base.tar", "model_state_dict"),
    "mask_head_hi.pth": ("next20_hires2/head_mid.tar", "model_state_dict"),
    "adapter_aug.pth": ("next13_general/adapter_aug.tar", "model_state_dict"),
    "adapter_aug_cg173.pth": ("next13_general/adapter_tta_from_aug.tar", "model_state_dict"),
    "depth_da_cg173.pth": ("next19_final2/adapter_tta_s0_n173.tar", "model_state_dict"),
    "depth_da_cg5.pth": ("next19_final2/adapter_tta_s0_n5.tar", "model_state_dict"),
    "student.pth": ("next26_kd3_coco/stage1.tar", "student"),
    "student_head_stage1.pth": ("next26_kd3_coco/stage1.tar", "head"),
    "student_depth.pth": ("next26_kd3_coco/stage2.tar", "depth"),
    "student_head.pth": ("next26_kd3_coco/stage2.tar", "head"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="checkpoints")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    for name, (rel, key) in MAP.items():
        s = os.path.join(a.src, rel)
        if not os.path.exists(s):
            print(f"  missing: {rel}")
            continue
        ck = torch.load(s, map_location="cpu", weights_only=False)
        torch.save({"model_state_dict": ck[key]}, os.path.join(a.dst, name))
        print(f"  {name:28s} <- {rel} [{key}]  {os.path.getsize(os.path.join(a.dst, name)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
