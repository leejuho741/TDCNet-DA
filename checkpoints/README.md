# Checkpoints

Weights are not stored in git. Download them from the release page (or the link in the main README) into this folder,
or, if you trained them yourself with the experiment scripts, run:

```bash
python scripts/export_checkpoints.py --src /content/drive/MyDrive/tdc_finetune --dst checkpoints
```

| File | Network | Used by |
|---|---|---|
| `tdcnet_repro.pth` | TDCNet, official recipe | baseline, `train_adapter.py --init` |
| `depth_da.pth` | TDCNet + DA injection (final depth model) | fast, accurate |
| `mask_head.pth` | mask head, Depth Anything at 252x336 | fast |
| `mask_head_hi.pth` | mask head, Depth Anything at 378x504 | accurate |
| `adapter_aug.pth` | frozen TDCNet + strong-augmentation adapter | generalization table |
| `depth_da_cg5.pth`, `depth_da_cg173.pth` | depth model adapted to ClearGrasp (5 / 173 images) | generalization table |
| `adapter_aug_cg173.pth` | strong-aug adapter adapted to ClearGrasp (173) | generalization table |
| `student.pth`, `student_depth.pth`, `student_head.pth` | distilled ResNet-18 pipeline | student |

Each file holds `{"model_state_dict": ...}`. The Depth Anything V2-Base encoder is downloaded from Hugging Face
(`depth-anything/Depth-Anything-V2-Base-hf`, CC-BY-NC-4.0) at first use.
