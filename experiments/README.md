# Experiment scripts

The original Google Colab scripts that produced every number in this repository, kept as they ran (comments and
log messages are in Korean). Each is a single self-contained cell: it mounts Google Drive, clones the needed
repositories, and reads/writes under `MyDrive/tdc_finetune/`. The cleaned implementation of the same logic is in
`tdc/` and `scripts/`; use those to reproduce.

| # | Script | What it does | Feeds |
|---|---|---|---|
| 01 | `01_tdcnet_repro.py` | TDCNet trained with the official recipe (fp32, official loader) | baseline 11.93 |
| 02 | `02_mask_head.py` | Mask heads on Depth Anything V2 Small / Base features | `mask_head.pth` |
| 03 | `03_da_adapter.py` | DA feature injection with TDCNet frozen (adapter) | 11.74 |
| 04 | `04_depth_da_scratch.py` | TDCNet + DA injection trained from scratch (seeds 0, 1) | final depth model 10.23 |
| 05 | `05_strong_aug_adapter_and_adaptation.py` | strong-augmentation adapter, first label-free adaptation | 17.9 on ClearGrasp |
| 06 | `06_adaptation_from_scratch_model.py` | adaptation of the final depth model (5 / 173 images) | 23.2 / 20.8 |
| 07 | `07_hires_mask_head.py` | 378x504 mask head | `mask_head_hi.pth` |
| 08 | `08_sota_quarter_test.py` | all methods, same setting, TransCG test 1/4 + ClearGrasp (zero-shot) | ClearGrasp column |
| 09 | `09_ablation.py` | ablation on the full test set | ablation / mask tables |
| 10 | `10_remake_adaptation.py` | ReMake with the same label-free adaptation | ReMake 5-image column |
| 11 | `11_distill_round1.py` | distillation, TransCG only | student 11.56 / 61.1 |
| 12 | `12_distill_round2_coco.py` | distillation with COCO photos, longer | student 11.31 / 50.7 |
| 13 | `13_figures.py` | README / slide figures | `assets/` |
| 14 | `14_speed_and_adapt_all_methods.py` | fp16 + CUDA graph speed of all methods; adaptation of TDCNet / FDCT / TODE / ReMake | speed, generalization table |
| 15 | `15_accurate_mode_tta.py` | accurate mode (hi-res head + TTA), test 1/4 | 10.15 / 31 FPS |
| 16 | `16_mask_sweep.py` | 1,120-setting mask post-processing sweep | Pareto check |
| 17 | `17_ablation_tta_speed.py` | ablation with and without TTA + speed per step | incremental ablation table |
| 18 | `18_student_speed.py` | student vs teacher speed, same setting | 68 / 85 FPS |
| 19 | `19_sota_full_test.py` | the comparison table on the full test set | comparison table |
| 20 | `20_gate_plugin.py` | our mask + gate added to FDCT / ReMake / TDCNet | plug-in table |
| 21 | `21_pred_figs.py` | depth predictions and error maps of one scene for each method | slide / README figures |
| 22 | `22_mask_figs.py` | GT and predicted masks of the same scene | slide figures |
| 23 | `23_prob_gate.py` | head probability used directly as the blend weight | mask post-processing table |
| 24 | `24_remake_resolution.py` | ReMake at 640x480 vs 320x240: accuracy and speed breakdown (T4) | ReMake 23.6 mm at 320 |
| 25 | `25_speed_same_resolution.py` | ours, TDCNet and ReMake timed in one run at the same resolution (T4) | same-GPU speed |
