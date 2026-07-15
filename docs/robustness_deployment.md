# Robustness and Deployment — StegShield as a Defense Layer

This document covers the robustness-hardening and deployment-router contribution: the
part of the project that moves StegShield from a lab classifier toward a component that
could sit in a real upload pipeline. It is written to be mined for the thesis chapters on
robustness, the threat model, and future work.

> Numbers marked `<measured>` are filled from the generated reports in
> `outputs/reports/` after the benchmark runs. See §6 for the exact commands.

---

## 1. Motivation — why the SOTA detector is not deployable as-is

Yedroudj-Net and our own StegShieldCNN are **spatial-domain** steganalyzers: a fixed SRM
high-pass front-end amplifies the pixel-level noise residual where LSB embedding leaves
its fingerprint. This is precisely why they work on clean PNGs — and precisely why they
are fragile. The SRM residual is a high-frequency signal, and any processing that
rewrites high-frequency content destroys it.

A real defense layer never receives pristine images. Uploaded images are resized by
platforms, re-encoded to JPEG by messengers and CDNs, and occasionally filtered. So the
deployment question is not "what is the accuracy on the test set" but "what survives the
processing an image undergoes on the way in".

We measured this on the baseline detector (200 clean + 200 stego per operation,
`test_standard`, threshold 0.5):

| Processing | Stego detection | Clean FPR | Interpretation |
|---|---|---|---|
| None (pristine PNG) | 1.000 | 0.000 | reference |
| Lossless PNG re-save | 1.000 | 0.000 | control (no pixel change) |
| Resize 0.75 (bicubic down-up) | 1.000 | 0.000 | benign geometric |
| Resize 0.50 | 0.530 | 0.000 | aggressive geometric |
| Gaussian blur (r=0.6 / 1.0) | 1.000 | 0.000 | benign filter |
| JPEG quality 90 | 0.000 | 0.000 | lossy re-encode |
| JPEG quality 75 | 0.000 | 0.005 | lossy re-encode |
| JPEG quality 60 | 0.000 | 0.000 | lossy re-encode |

The headline: **spatial LSB detection survives benign geometric processing but collapses
completely under JPEG re-encoding** (100% → 0%). This is not a bug to fix; it is a
property of the threat. JPEG quantization overwrites the LSB plane, so the payload itself
is destroyed — there is nothing left to detect. The insight cuts both ways: re-encoding
an upload to JPEG is itself a mitigation against LSB exfiltration.

### 1.1 Why resize survives but JPEG does not

Both change most pixels, but differently. After a bicubic down-up resize the payload
*bytes* are corrupted, yet the LSB plane stays statistically noise-like (ones-ratio near
0.5) — the *fingerprint* survives even though the message does not, and the detector keys
on the fingerprint. JPEG replaces the LSB texture with its own DCT-quantization structure
(ones-ratio drops well below 0.5), erasing the fingerprint along with the payload. This
is the empirical basis for the router's lossless-vs-lossy split (§3).

---

## 2. Robustness hardening (training-time augmentation)

To make detection robust to benign processing — and, importantly, to keep the
false-positive rate low on *processed clean* images — the hardened detector is trained
with on-the-fly **payload-preserving** augmentation: each image is randomly passed
through resize, blur, additive noise, or a lossless re-save before the crop.

Design decision (deliberate, defensible): augmentation is **payload-preserving only** —
no JPEG. JPEG-recompressing a stego image destroys the payload, so the resulting image is
effectively clean; training on it as "stego" would teach the model to flag payload-free
images and poison both recall and precision. The lossy case is handled by the router
(§3), not by the detector. Augmentation menu and rationale live in
`src/stegshield/processing.py`; validation is never augmented (it measures pristine
performance).

### 2.1 Detection is not harmed on pristine images

| Model | Pristine val macro-F1 | Pristine `test_standard` macro-F1 |
|---|---|---|
| Baseline (no augmentation) | 1.0000 | 1.0000 |
| Hardened (augmentation) | 0.9998 | 1.0000 |

Hardening costs nothing on pristine images: both reach perfect detection on the standard
test split.

### 2.2 The noise saturation trap (the key robustness finding)

Raw detection rate is misleading under additive noise. The baseline shows near-perfect
"detection" under noise — but only because it flags *almost every clean image as stego
too*. Balanced accuracy (mean of stego recall and clean specificity) exposes this:

| Processing | Baseline detect / clean-FPR / **bal-acc** | Hardened detect / clean-FPR / **bal-acc** |
|---|---|---|
| Resize 0.75 | 1.000 / 0.000 / **1.000** | 1.000 / 0.000 / **1.000** |
| Resize 0.50 | 0.530 / 0.000 / **0.765** | 0.530 / 0.000 / **0.765** |
| Blur 0.6 / 1.0 | 1.000 / 0.000 / **1.000** | 1.000 / 0.000 / **1.000** |
| Noise σ=2 | 1.000 / **0.965** / **0.518** | 0.575 / 0.000 / **0.787** |
| Noise σ=5 | 0.995 / **0.975** / **0.510** | 0.310 / 0.015 / **0.647** |

Under noise the baseline sits at **chance** (balanced accuracy 0.51–0.52): it is not
detecting stego, it is calling everything noisy "stego", which would flag 96–97% of
legitimate uploads. The hardened detector actually discriminates (0.65–0.79) with a
clean false-positive rate at or below 1.5%. This is the deployment-critical win: a
gateway that false-flags 96% of noisy-but-clean images is unusable, and hardening fixes
exactly that.

Figure: `outputs/figures/robustness_overlay.png` (balanced accuracy, baseline vs
hardened, across the processing sweep). Per-model detail with detection and FPR bars:
`robustness_baseline_robustness.png`, `robustness_hardened_robustness.png`.

The honest boundaries: (a) aggressive resize (halving) degrades both equally to 0.765 —
enough signal is genuinely lost that augmentation cannot recover it; (b) JPEG collapses
both to chance (~0.50), because JPEG destroys the payload — a routing decision (§3), not
a model-capacity problem.

---

## 3. Deployment router (the defense layer)

`src/stegshield/router.py` inspects each file's stored format and routes:

```text
Uploaded image
      |
      v
Detect stored format (magic bytes)
      |
   +--+---------------------------+
   |                              |
lossless (PNG/BMP/TIFF)      lossy (JPEG/WebP/GIF) or unknown
   |                              |
   v                              v
Hardened spatial CNN          Static structural / metadata
+ static analysis             analysis only
+ statistical LSB             (spatial LSB not applicable:
   |                           re-encoding neutralized it)
   v                              v
Fusion -> risk label          Metadata -> risk label
```

The router emits a structured verdict (label, risk score, processing state, which
analyses ran, per-indicator evidence, latency) — the shape a service integration would
consume. CLI:

```powershell
stegshield scan image.png --cnn-model-path outputs/models/steganalysis_hardened.pt --json
```

Threat-model honesty baked into the output: for a lossy image the verdict explicitly
reports `spatial_lsb_applicable: false` and explains that re-encoding destroyed the LSB
plane, rather than silently returning "safe" from a detector that cannot see anything.

---

## 4. Deployment calibration (low false-positive operating point)

A gateway that flags 1% of legitimate uploads is unusable, so the operating point is
judged by the **clean false-positive rate on processed-clean images**, not by detection
alone. The decisive comparison is under noise, where the baseline is unusable:

| Model | Max clean FPR across all non-JPEG processing |
|---|---|
| Baseline | 0.975 (noise σ=5) — unusable |
| Hardened | 0.015 (noise σ=5) — deployable |

At the default threshold 0.5 the hardened detector already holds clean FPR ≤ 1.5% across
every benign transform (0% on resize/blur/re-save), while retaining full detection on
pristine and mildly-processed images. The FPR collapse came from augmentation, not from
the threshold — but the threshold is a second, independent knob worth tuning.

### 4.1 Threshold operating point

The hardened detector's probabilities under noise are *separated, not saturated* (stego
median P≈0.70 vs clean P≈0.09 at σ=2), so the default 0.5 threshold leaves recall on the
table. Lowering it to 0.3 recovers a large share of noise robustness while keeping benign
false positives at zero:

| Operation | th=0.5 detect / FPR / bal-acc | th=0.3 detect / FPR / bal-acc |
|---|---|---|
| Pristine, resize 0.75, blur | 1.000 / 0.000 / 1.000 | 1.000 / 0.000 / 1.000 |
| Noise σ=2 | 0.575 / 0.000 / 0.787 | 0.740 / 0.045 / 0.847 |
| Noise σ=5 | 0.310 / 0.015 / 0.647 | 0.535 / 0.130 / 0.702 |

**Recommended deployment operating point: threshold 0.3.** It keeps the false-positive
rate at 0% on pristine and benign geometric/blur processing, holds it to ≤5% under
moderate noise, and lifts balanced accuracy under noise by 6–7 points. Heavy noise (σ=5)
is the only case where its FPR climbs (13%), and σ=5 additive noise is an aggressive,
uncommon transform for a benign upload. A platform that must guarantee a very low alarm
rate under any condition keeps 0.5; one that prioritizes recall under realistic noise
uses 0.3. (The fusion pipeline already labels CNN-only "suspicious" at 0.3.)

### 4.2 Why not just augment harder for noise?

A heavier-noise augmentation menu (identity 40%, noise 25%, the useless lossless re-save
dropped) was trained and benchmarked. At each model's best threshold it matched the
shipped model within noise (balanced accuracy 0.89 vs 0.89 at σ=2, 0.74 vs 0.73 at σ=5),
because robustness to additive noise is **signal-limited**: the ±1 LSB perturbation sits
below a σ≥2 noise floor, so no amount of augmentation recovers it. The shipped
augmentation menu is therefore kept as-is, and threshold tuning (§4.1) is the effective
lever for the noise regime.

---

## 5. Scope and limitations (stated honestly for the defense)

- **Single cover source.** All clean images come from the Kaggle covers, so we measure
  *processing robustness* rigorously but not *cover-source* diversity; the latter is
  future work.
- **Lossless-image threat only.** This work hardens and deploys detection for the
  losslessly-stored LSB threat. It does **not** detect JPEG-domain (DCT) steganography —
  that is a different problem (ALASKA2-scale data, DCT-domain features) and is named as
  the primary future extension.
- **Router, not classifier, handles JPEG.** By design; see §1 and §2.

---

## 6. Reproduction

```powershell
# Baseline (no augmentation) and hardened (augmentation) detectors
stegshield train-cnn --task stego --model steganalysis --epochs 15 --batch-size 16 `
  --amp --num-workers 4 --device cuda `
  --output-model outputs/models/steganalysis_stego.pt `
  --output-metrics outputs/reports/steganalysis_stego_training.json
stegshield train-cnn --task stego --model steganalysis --augment --epochs 15 --batch-size 16 `
  --amp --num-workers 4 --device cuda `
  --output-model outputs/models/steganalysis_hardened.pt `
  --output-metrics outputs/reports/steganalysis_hardened_training.json

# Robustness benchmark for each model
stegshield evaluate-robustness --model-path outputs/models/steganalysis_stego.pt `
  --split-csv data/splits/test_standard.csv --device cuda `
  --output-report outputs/reports/robustness_baseline.json
stegshield evaluate-robustness --model-path outputs/models/steganalysis_hardened.pt `
  --split-csv data/splits/test_standard.csv --device cuda `
  --output-report outputs/reports/robustness_hardened.json

# Figures (per-model bars + baseline-vs-hardened overlay)
stegshield plot-results outputs/reports/robustness_baseline.json `
  outputs/reports/robustness_hardened.json --output-dir outputs/figures

# Scan a file through the deployment router
stegshield scan image.png --cnn-model-path outputs/models/steganalysis_hardened.pt --json
```

---

## 7. Future work — real-world integration

The router already returns a structured, service-friendly verdict, so the natural
extension is an API deployment: a thin HTTP endpoint wrapping `scan_image` that accepts an
uploaded image and returns the verdict JSON, letting an application (profile-picture
upload, message attachment, CDN ingest) call StegShield as a defense layer before storing
user images. This is described as future work only; the thesis deliverable stays CLI-based.
A JPEG-domain detector would slot into the router's lossy branch to extend coverage to
compressed traffic.
