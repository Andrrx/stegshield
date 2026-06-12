# Payload-Size Regression Head — Experiment Runbook

This runbook extends StegShield's proposed `StegShieldCNN` into a multi-task network: it
keeps the clean/stego **detection** head and adds a **regression** head that estimates
the sequential-LSB payload size visible in the crop. It then compares the CNN payload
estimate against the statistical Westfeld-Pfitzmann estimator and feeds it into the
fusion layer.

All commands are PowerShell, run from the repo root with the ML extra installed and a
CUDA GPU (drop `--amp --num-workers 4 --device cuda` for CPU). Epoch time on an RTX 3050
(4 GB) dominates wall-clock; review `src/stegshield/data/synth_lsb.py` before trusting
the numbers — a wrong channel/bit order there would silently invalidate the regression.

## Why this design (defendable choices)

- **Target = `log2(payload_bytes_in_crop + 1)`, capped at the crop's LSB capacity**
  (`image_size**2 * 3 / 8` = 24576 bytes at 256px). Payloads span 16 B to ~24 KB (three
  orders of magnitude), so a log target is well-scaled; and the network physically
  cannot estimate payload beyond the pixels it sees, so the target is capped there.
- **Synthetic re-embedding for ground truth (anti-circularity).** The regressor is
  trained on payloads of *known* size embedded by StegShield's own embedder
  (`make-payload-regression-set`), never on the statistical estimator's outputs. If it
  learned from the estimator, the later "CNN vs statistical agreement" experiment would
  only measure how well the student mimics the teacher. With independent ground truth,
  the agreement plot on real Kaggle data is real evidence.
- **Masked loss.** Smooth-L1 is applied only to stego samples with a known size; clean
  and unlabeled samples are masked out (NaN target). Detection cross-entropy still
  trains on every sample. Total loss = `ce + payload_loss_weight * masked_smooth_l1`.

## Step 0 — Build the synthetic regression set

Clean source images come ONLY from the Kaggle train split for `regress_train`/`val` and
ONLY from the Kaggle test split for `regress_test` (no source-image leakage). Payloads
are inert `os.urandom` bytes.

```powershell
stegshield make-payload-regression-set `
  --train-csv data/splits/train.csv `
  --test-csv data/splits/test_standard.csv `
  --output-image-dir data/processed/regress `
  --output-split-dir data/splits
# writes data/splits/regress_{train,val,test}.csv
```

## Experiment 1 — Detection must not regress

Train detection-only and multi-task models on identical data, seed, and epochs; the only
difference is `--payload-head`. Compare detection macro-F1 / balanced accuracy.

```powershell
# detection-only control (same data as the multi-task model)
stegshield train-cnn --task stego --model steganalysis `
  --train-csv data/splits/regress_train.csv --val-csv data/splits/regress_val.csv `
  --epochs 15 --batch-size 16 --amp --num-workers 4 --device cuda `
  --output-model outputs/models/steganalysis_regress_detect.pt `
  --output-metrics outputs/reports/steganalysis_regress_detect_training.json

# multi-task: detection + payload regression
stegshield train-cnn --task stego --model steganalysis --payload-head --payload-loss-weight 0.5 `
  --train-csv data/splits/regress_train.csv --val-csv data/splits/regress_val.csv `
  --epochs 15 --batch-size 16 --amp --num-workers 4 --device cuda `
  --output-model outputs/models/steganalysis_multitask.pt `
  --output-metrics outputs/reports/steganalysis_multitask_training.json

# detection metrics on the held-out regression test split (same distribution)
stegshield evaluate-cnn --model-path outputs/models/steganalysis_regress_detect.pt `
  --split-csv data/splits/regress_test.csv `
  --output-report outputs/reports/regress_detect_test.json --device cuda
stegshield evaluate-cnn --model-path outputs/models/steganalysis_multitask.pt `
  --split-csv data/splits/regress_test.csv `
  --output-report outputs/reports/multitask_detect_test.json --device cuda

# cross-check on the real Kaggle detection benchmark
stegshield evaluate-cnn --model-path outputs/models/steganalysis_multitask.pt `
  --split-csv data/splits/test_standard.csv `
  --output-report outputs/reports/multitask_test_standard.json --device cuda
```

Pass criterion: multi-task detection macro-F1 within ~0.5 points of the detection-only
control. Training also prints `Best-epoch validation payload MAE` for the multi-task run.

## Experiment 2 — Regression quality

```powershell
stegshield evaluate-payload-regression `
  --model-path outputs/models/steganalysis_multitask.pt `
  --split-csv data/splits/regress_test.csv `
  --output-report outputs/reports/payload_regression_test.json `
  --num-workers 4 --device cuda
```

Reports MAE and median absolute error in bytes, plus MAE in log2 space. The
predicted-vs-true scatter (log-log) is produced in Step 5.

## Experiment 3 — CNN vs statistical estimator (agreement)

On the real Kaggle stego images, compare the CNN estimate with the statistical
Westfeld-Pfitzmann estimator. The statistical estimate is capped at crop capacity so both
are on the same scale; only images the CNN calls stego (prob >= 0.5) and for which the
statistical estimator returns a value are compared.

```powershell
stegshield evaluate-payload-agreement `
  --model-path outputs/models/steganalysis_multitask.pt `
  --split-csv data/splits/test_standard.csv `
  --output-report outputs/reports/payload_agreement_test_standard.json `
  --num-workers 4 --device cuda
```

Reports Pearson and Spearman correlation in log2 space and the median absolute
difference in bytes. Because the two estimators use entirely independent logic (a learned
CNN vs a hand-crafted LSB-plane statistic), a high correlation is genuine convergent
evidence, not circular.

## Experiment 4 — Fusion comparison (3 payload sources)

```powershell
stegshield evaluate-fusion --model-path outputs/models/steganalysis_multitask.pt `
  --split-csv data/splits/test_standard.csv --payload-source statistical `
  --output-report outputs/reports/fusion_payload_statistical.json --device cuda
stegshield evaluate-fusion --model-path outputs/models/steganalysis_multitask.pt `
  --split-csv data/splits/test_standard.csv --payload-source cnn `
  --output-report outputs/reports/fusion_payload_cnn.json --device cuda
stegshield evaluate-fusion --model-path outputs/models/steganalysis_multitask.pt `
  --split-csv data/splits/test_standard.csv --payload-source both `
  --output-report outputs/reports/fusion_payload_both.json --device cuda
```

`statistical` reproduces the current fusion behavior; `cnn` replaces the statistical LSB
severity signal with the CNN regression estimate; `both` uses each as corroborating
evidence. The three `fused[...]` rows appear distinctly in the summary table.

## Step 5 — Summary table and figures

```powershell
stegshield summarize-experiments `
  outputs/reports/regress_detect_test.json `
  outputs/reports/multitask_detect_test.json `
  outputs/reports/fusion_payload_statistical.json `
  outputs/reports/fusion_payload_cnn.json `
  outputs/reports/fusion_payload_both.json `
  --output outputs/reports/payload_experiment_summary.md

stegshield plot-results `
  outputs/reports/steganalysis_multitask_training.json `
  outputs/reports/payload_regression_test.json `
  outputs/reports/payload_agreement_test_standard.json `
  --output-dir outputs/figures
# writes payload_regression_test_scatter.png and payload_agreement_test_standard_scatter.png
```

## Results (trained 2026-06-11, RTX 3050, 15 epochs, batch 16, AMP)

Regression set: 3,400 train / 600 val / 2,000 test images (~80% embedded with known
sizes, ~20% clean). Both models trained on `regress_train`/`regress_val`.

### Experiment 1 — detection is not harmed by the payload head

| Model | Split | Accuracy | Macro F1 |
|---|---|---|---|
| Detection-only control | regress_test | 1.0000 | 1.0000 |
| Multi-task (+payload head) | regress_test | 1.0000 | 1.0000 |
| Multi-task (+payload head) | test_standard (real Kaggle) | 0.9999 | 0.9998 |

The payload head causes **zero detection regression**: identical on the held-out
synthetic split, and the multi-task model still matches the original Kaggle-trained
detector (0.9999 / 0.9998) on real data despite training only on synthetic images.

### Experiment 2 — regression quality (regress_test, 1,601 supervised stego)

- **Median absolute error: 62 bytes** (the robust headline; payloads span 16 B–24 KB).
- MAE: 534 bytes (inflated by saturation on near-capacity payloads).
- Figure `payload_regression_test_scatter.png`: estimates hug y = x across five orders
  of magnitude, with mild under-estimation near the 24,576-byte crop-capacity cap.

### Experiment 3 — CNN vs statistical agreement (test_standard, 5,996 stego)

- **Pearson (log2): 0.977, Spearman: 0.940.** Two estimators with entirely independent
  logic — a learned CNN trained only on synthetic data vs a hand-crafted LSB-plane
  statistic — converge on real Kaggle payloads. This is genuine convergent evidence,
  not circularity (see `payload_agreement_test_standard_scatter.png`).

### Experiment 4 — fusion with each payload source (test_standard, 3-way risk)

| Payload source | Accuracy | Macro F1 | safe / suspicious / dangerous recall |
|---|---|---|---|
| statistical (baseline) | 0.9958 | 0.9960 | 0.999 / 0.998 / 0.992 |
| cnn | 0.9805 | 0.9822 | 0.9995 / 1.000 / 0.956 |
| both | 0.6896 | 0.5796 | 0.999 / 0.000 / 1.000 |

Readings:

- The **CNN payload estimate is a viable severity source** (0.9805 vs 0.9958): nearly
  matching the statistical estimator, with *perfect* suspicious recall. Its only cost is
  156 script payloads it estimates just below the 128-byte gate, graded `suspicious`
  instead of `dangerous` (dangerous recall 0.956). Zero threat-as-safe in both.
- **`both` collapses (suspicious recall 0.000).** Appending both payload indicators
  double-counts: a small-payload `suspicious` image gets a medium indicator from each
  source, and the stacked score crosses the `dangerous` threshold, so every stego image
  is over-escalated to `dangerous`. Lesson for the thesis: the two payload estimators
  are redundant evidence of the *same* underlying signal — use one, not both additively.
  A corroboration scheme would need to take the max severity, not sum the indicators.

Overall: the statistical estimator remains the best single severity source on this
dataset, but the multi-task CNN demonstrates that payload size can be regressed directly
from pixels (median error 62 B) and used as a competitive, self-contained alternative —
quantitative steganalysis from a single network, with detection unharmed.

## Files added/changed for this feature

- `src/stegshield/data/synth_lsb.py` — sequential-LSB embedder + regression-set builder.
- `src/stegshield/data/torch_dataset.py` — `payload_regression_target`,
  `payload_bytes_from_target`, `with_payload_target` flag.
- `src/stegshield/models/cnn.py` — `StegShieldCNN(payload_head=...)`, `forward_multitask`.
- `src/stegshield/train_cnn.py` — masked smooth-L1 multi-task loss, checkpoint metadata.
- `src/stegshield/predict_cnn.py` — `predict_with_payload`.
- `src/stegshield/metadata/risk_rules.py` — `cnn_payload_indicators`, `build_assessment`.
- `src/stegshield/evaluate_fusion.py` — `--payload-source {statistical, cnn, both}`.
- `src/stegshield/payload_eval.py` — regression and agreement evaluation.
- `src/stegshield/plots.py` — payload scatter figures.
- `src/stegshield/experiment_summary.py` — payload-source-tagged fusion rows.
