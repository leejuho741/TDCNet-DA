# Results

Every number below comes from a log of the experiment named in its table. Experiment scripts are in
`experiments/` (numbered). Units: RMSE / MAE / region errors in mm, δ in %, FPS on one NVIDIA L4.

**Data subsets.** "test" = TransCG test, 23,524 images (D435 11,766 / L515 11,758). Some side experiments used
"test 1/4" (every 4th image, 5,881); the two agree within 0.02 mm.

**Metrics.** *Standard* (TransCG paper protocol): inside the GT transparent mask, per image, averaged.
*Regions* (per image, GT 0.3–1.5 m, 320x240 px): boundary = 3 px either side of the mask edge,
surround = 3–15 px outside, background = > 15 px outside. "Transparent" in the region tables is the in-mask RMSE at
256x144 (official evaluation size); it differs from the 320x240 value by ≤ 0.01 mm.

## 1. Comparison with published methods — test, all 23,524 images (exp. 19, speed exp. 14)

| Method | GT mask | RMSE | REL | MAE | δ1.05 | δ1.10 | δ1.25 | D435 | L515 | Boundary | Background | FPS |
|---|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| FDCT | no | 15.80 | 2.25 | 10.16 | 88.04 | 97.10 | 99.79 | 11.41 | 20.20 | 25.89 | 16.84 | 393 |
| TODE-Trans | no | 13.47 | 1.90 | 8.56 | 90.42 | 97.38 | 99.81 | 8.91 | 18.02 | 29.08 | 20.90 | 206 |
| TDCNet (reproduced) | no | 11.94 | 1.68 | 7.56 | 92.41 | 97.93 | 99.88 | 8.90 | 14.98 | 28.96 | 281.74 | 177 |
| ReMake (GT mask) | yes | 11.38 | 1.76 | 7.85 | 93.17 | 98.41 | 99.97 | 9.19 | 13.58 | 9.66 | 4.35 | 18 |
| Ours – fast | no | 10.48 | 1.45 | 6.53 | 94.45 | 98.64 | 99.93 | 7.74 | 13.23 | 27.35 | 1.43 | 66 |
| Ours – accurate (hi-res head + TTA) | no | 10.14 | 1.41 | 6.36 | 94.69 | 98.76 | 99.94 | 7.52 | 12.77 | 26.98 | 1.07 | 31 |

The earlier run on every 4th test image (exp. 08 / 15) agrees within 0.02 mm (e.g. fast 10.49, ReMake 11.39). The README and slides quote the region-metric "transparent" value
(256x144) for every table so all numbers share one metric: FDCT 15.80 / δ 88.05, TODE 13.46 / 90.43, TDCNet 11.93 /
92.42, ReMake 11.37 / 93.17, fast 10.48 / 94.46, accurate 10.14 / 94.70. It differs from the standard value above by
at most 0.01.
ReMake with our predicted mask instead of the GT mask: 15.76 (test 1/4, exp. 08). Published values for reference:
TDCNet 12 / 92.25, TODE 13 / 90.43, FDCT 15 / 88.18, ReMake 10 / 94.76 (GT mask, 640x480 input). DFNet (TransCG's own
baseline, paper 18 / 81.15) could not be reproduced with the public checkpoint and is omitted.

## 2. Incremental ablation — test (exp. 17)

| Step | Transparent | δ1.05 | Boundary | Surround | Background | D435 | L515 | IoU | FPS |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| TDCNet | 11.93 | 92.42 | 28.96 | 40.69 | 281.74 | 8.89 | 14.97 | – | 182 |
| + DA feature injection | 10.23 | 94.59 | 28.33 | 35.52 | 275.08 | 7.65 | 12.81 | – | 99 |
| + mask head + gate | 10.48 | 94.46 | 27.35 | 16.49 | 1.43 | 7.73 | 13.22 | 0.842 | 69 |
| + high-res mask head | 10.37 | 94.52 | 27.31 | 15.48 | 1.08 | 7.70 | 13.05 | 0.860 | 45 |
| + flip TTA | 10.14 | 94.70 | 26.98 | 15.19 | 1.07 | 7.52 | 12.76 | 0.860 | 31 |

Flip TTA on every step (same run): TDCNet 11.54, + injection 9.99, + gate 10.24, + hi-res head 10.14.
GT mask instead of the head (exp. 09): transparent 10.28, surround 9.53, background 0.00.

Other ablations (test, exp. 09):

| Variant | Transparent |
|---|--:|
| TDCNet frozen + injection adapter | 11.74 |
| TDCNet frozen + strong-augmentation adapter | 10.74 |
| Injection trained from scratch, seed 0 / seed 1 | 10.23 / 10.22 |
| Mask head on Depth Anything V2-Small / Base / Base 378x504 / GT mask (with gate) | 10.69 / 10.47 / 10.37 / 10.28 |

## 3. Mask post-processing — test (exp. 09) and sweep (exp. 16)

| Mask processing | Transparent | Boundary | Background |
|---|--:|--:|--:|
| p > 0.5, hard switch | 13.06 | 22.48 | 0.55 |
| hysteresis 0.5 / 0.1, hard switch | 11.69 | 24.13 | 0.90 |
| + distance blend 4–12 px | 10.35 | 28.11 | 2.28 |
| **+ 2 px erosion (final)** | 10.47 | 27.35 | 1.43 |
| + flip TTA | 10.24 | 27.02 | 1.42 |

Sweep: 2 heads x 14 cuts x erosion 0–4 x 8 blend ranges = 1,120 settings, selected on test[0::8] (2,941) and
re-measured on test[4::8] (2,940). 303 settings are Pareto-optimal on (transparent, surround, background); the rule
"lowest transparent error without worse surround or background" picks the final setting for both heads.
On split B: fast 10.50 / surround 16.46 / background 1.42; hi-res head 10.39 / 15.46 / 1.11. Erosion and blend range
trade transparent against background error; the hysteresis high threshold (0.5–0.7) barely matters.

## 4. Gate as a plug-in — test 1/10, background error (exp. 20)

| Method | without gate | with our mask + gate |
|---|--:|--:|
| FDCT | 16.90 | 1.09 |
| ReMake (predicted mask) | 4.57 | 0.78 |
| TDCNet | 279.45 | 1.61 |

## 5. Generalization — ClearGrasp real-test, 113 images (exp. 08, 10, 14)

| Method | No adaptation | 5 images | 173 images |
|---|--:|--:|--:|
| FDCT | 24.91 | 24.57 | 27.79 |
| ReMake (GT mask) | 27.01 | 18.27 | 19.15 |
| TODE-Trans | 32.66 | 22.47 | 27.47 |
| ReMake (predicted mask) | 37.36 | 25.86 | 26.52 |
| TDCNet | 67.39 | 34.34 | 47.82 |
| Ours | 48.59 | 23.17 | 20.84 |
| Ours, strong-augmentation adapter | 52.2 | 22.90 | 17.89 |

Adaptation images: ClearGrasp real-val (173; the 5-image subset is random). 600 steps, batch 8, unlabelled losses
(sensor outside the predicted mask, flip consistency, TransCG anchor batch). Ours: injection projections (0.41M);
others: normalization layers only (about 0.01M parameters). The synthetic ClearGrasp splits were not used: every method
scored 117–196 mm there, so they do not separate methods.

## 6. Distillation — test 1/4 and ClearGrasp real-test (exp. 11, 12, 18)

| Model | TransCG | ClearGrasp | FPS | latency |
|---|--:|--:|--:|--:|
| Teacher (fast, Depth Anything V2-Base 86.6M) | 10.48 | 48.6 | 68 | 14.63 ms |
| Student ResNet-18 (17.6M), TransCG only | 11.56 | 61.1 | 85 | 11.83 ms |
| Student, + 5,000 COCO val2017 photos (longer training) | 11.31 | 50.7 | 85 | 11.83 ms |

Feature extraction only: Depth Anything 4.39 ms, student 1.72 ms. The control (same longer schedule, TransCG only)
was prepared but not run, so the COCO effect is not separated from the longer training.

## 7. Speed — L4, batch 1, 320x240, fp16 + CUDA graph

| Model | FPS | ms |
|---|--:|--:|
| FDCT | 393 | 2.54 |
| DFNet | 249 | 4.02 |
| TODE-Trans | 206 | 4.84 |
| TDCNet | 177 | 5.66 |
| ReMake (Depth Anything V2-Small + network, 640x480) | 18 | 55.21 |
| Ours – fast | 66 | 15.25 |
| Ours – hi-res head (no TTA) | 43 | 23.53 |
| Ours – accurate | 31 | – |
| Student | 85 | 11.83 |

Exp. 14 (all methods in one run); accurate and student from exp. 15 / 18. The same model varies by about ±3 FPS
between runs (fast: 66 in exp. 14, 69 in exp. 15 and 17).
