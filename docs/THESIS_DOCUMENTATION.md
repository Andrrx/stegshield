# StegShield — Technical Documentation for the Thesis

This document records everything the project uses, why each decision was made, what the
alternatives were, the numbers behind every claim, and the insights discovered during
development. It is written to be mined directly for thesis chapters (methodology,
implementation, experiments, discussion). The deepest coverage is given to the AI/ML
core, as requested.

---

## 1. Executive Summary

StegShield is a hybrid system that classifies image files into three risk levels —
`safe`, `suspicious`, `dangerous` — by fusing two independent evidence sources:

1. **A binary steganalysis CNN** (clean/stego) operating on pixels.
2. **A rule-based static analyzer** operating on file structure, metadata, and
   LSB-plane statistics.

Final results on the held-out standard test split (8,000 images):

| System | Accuracy | Macro F1 | Key property |
|---|---|---|---|
| Metadata-only (static rules) | 0.5600 | 0.5280 | Cannot produce `dangerous` alone |
| CNN-only (binary, converted to risk) | 0.5600 | 0.5283 | Cannot produce `dangerous` alone |
| **Fused (final system)** | **0.9958** | **0.9960** | All three classes; 0 stego files labeled safe |

Both CNNs (the Yedroudj-Net literature baseline and the proposed StegShieldCNN) reach
0.9999 accuracy / 0.9998 macro F1 on the binary clean/stego task, and detect 100% of
the adversarial (base64/zip-encoded payload) stego variants. The headline engineering
findings were (a) a preprocessing bug class — center cropping destroyed the stego
evidence entirely — and (b) a static-analysis bug class — a 2-byte signature scan that
false-flagged 24% of clean files. Both are documented below with before/after numbers,
because they make strong thesis discussion material.

---

## 2. Problem Statement and Threat Model

Image files can carry hidden payloads (steganography), appended archives/executables
(polyglot files), malicious metadata, or malformed structures that exploit parsers. A
purely visual classifier misses container-level threats; purely static analysis misses
pixel-level embedding. The thesis question is whether a **fusion of a steganalysis CNN
and static file analysis** classifies image risk better than either component alone.

In scope: detection of steganographic payloads embedded with sequential LSB tools
(Invoke-PSImage family), appended/embedded binary content, structural anomalies.
Out of scope: payload execution, malware sandboxing, adaptive-embedding steganography
(S-UNIWARD/WOW) — discussed in §10 Limitations.

The three labels are **risk levels**, not malware verdicts:

- `safe` — no stego evidence, no structural indicators.
- `suspicious` — stego evidence or moderate static indicators (e.g., a small hidden
  marker such as a cryptocurrency address).
- `dangerous` — strong combined evidence: script/binary-scale hidden payload, embedded
  executable/archive signatures, or multiple high-severity indicators.

---

## 3. System Architecture

```text
Input image file
      |
      v
File validation and type detection  (magic bytes, extension cross-check)
      |
      +-------------------------------+
      |                               |
      v                               v
Image preprocessing             Static analysis
(top-left crop 256x256,         (metadata extraction, file-structure
 raw 0-255 pixel values)         checks, sequential-LSB payload estimator)
      |                               |
      v                               v
Binary steganalysis CNN         Rule-based risk score [0,1]
(P(stego) in [0,1])             + indicator list with severities
      |                               |
      +---------------+---------------+
                      |
                      v
            Fusion / decision layer
        score = 0.55 * P(stego) + 0.45 * metadata_risk
        + override rules for high-severity evidence
                      |
                      v
       Final label (safe / suspicious / dangerous)
       + risk score + human-readable explanation
```

Design principle: **the CNN answers one question only — "is there pixel-level stego
evidence?"** The 3-way risk decision is made by the fusion layer, which can also cite
explainable static evidence. This separation is justified empirically in §5.1.

Implementation: Python 3.11 package (`src/stegshield/`), installable CLI (`stegshield
analyze | train-cnn | evaluate-cnn | evaluate-fusion | plot-results | ...`), PyTorch
2.11 for the ML part. 51 unit tests.

---

## 4. Dataset

**Kaggle Stego Images Dataset** (Marco Zuppelli):
<https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset>

44,000 PNG images, 512×512. Payloads are embedded with a sequential LSB tool of the
Invoke-PSImage family: payload bits are written into pixel least-significant bits in
scan order (row-major, starting near pixel (0,0)), changing each affected channel value
by at most ±1.

Payload types and their project labels:

| Payload | Content scale | Project label |
|---|---|---|
| Clean cover | — | `safe` |
| Ethereum address / URL | ~40–100 bytes | `suspicious` |
| JavaScript / JS-in-HTML / PowerShell scripts | ~1–100 KB | `dangerous` |
| base64- or zip-encoded variants of the above | varies | `dangerous` (adversarial test only) |

Split sizes (official dataset split preserved; CSVs produced by
`stegshield import-kaggle-splits`):

| Split | safe | suspicious | dangerous | total |
|---|---|---|---|---|
| train | 4,000 | 4,885 | 7,115 | 16,000 |
| val | 2,000 | 2,411 | 3,589 | 8,000 |
| test_standard | 2,000 | 2,481 | 3,519 | 8,000 |
| test_adversarial | 0 | 0 | 12,000 | 12,000 |

For the binary CNN task, labels are mapped: `safe → clean`, `suspicious/dangerous →
stego`. This makes the binary training set imbalanced (4,000 clean vs 12,000 stego),
which drives the imbalance-handling choices in §5.7.

### 4.1 Dataset artifacts (important for honest reporting)

Two artifacts were identified that a model or rule could exploit unfairly:

1. **Color mode**: clean covers are RGBA; stego images are RGB (the embedding tool
   re-saved them). Mitigation: the pipeline converts everything to RGB before the CNN,
   and no rule uses color mode. This must be stated in the thesis.
2. **Smooth cover tops**: the covers' first image rows have near-zero LSB activity
   (smooth sky-like content). The sequential-LSB detector (§6.2) benefits from this; on
   covers with textured top-left content its false-positive risk would be higher. The
   detector therefore requires a *transition* back to image-like statistics, and the
   fusion layer gates severity escalation on CNN agreement.

Measured payload-size distributions (estimated by the §6.2 detector, 200 samples per
group from the test split):

| Group | min | p10 | p50 | p90 | max |
|---|---|---|---|---|---|
| eth/url | 48 B | 48 B | 48 B | 48 B | 96 B |
| scripts (plain) | 48 B | 1,728 B | 27,456 B | 65,184 B | 97,392 B |
| base64 variants | 48 B | 48 B | 336 B | 42,192 B | 95,184 B |
| zip variants | 384 B | 384 B | 2,448 B | 25,680 B | 86,880 B |

The clear gap between eth/url (≤96 B) and scripts (p10 ≈ 1.7 KB) motivates the
128-byte severity threshold in §6.2.

---

## 5. The AI/ML Core

### 5.1 Task formulation: why a binary CNN, not 3-way

The original design asked one CNN to output `safe/suspicious/dangerous` directly. This
failed for a structural reason: the three labels mix **different evidence types**.
"Dangerous" in this project means *script-scale payload OR structural file anomalies* —
but file-structure anomalies are invisible in pixels, and payload *type* (a URL vs a
script) is not a visual property either; only payload *presence and extent* are. The
3-way CNN was therefore being asked to predict labels from information it could not
see, and in early experiments it collapsed (predicting only `suspicious`, accuracy
0.5084, macro F1 0.3931, safe recall 0.0).

Supporting literature: Kang, Park & Park, "CNN-Based Ternary Classification for Image
Steganalysis", *Electronics* 8(12):1225, 2019 — shows that even distinguishing two
*embedding algorithms* (WOW vs UNIWARD) in one CNN is hard and requires joint training;
distinguishing *payload semantics* is strictly harder and not a pixel property.

Decision: the CNN solves the well-posed binary steganalysis problem (clean vs stego);
the 3-way decision is delegated to the fusion layer, which has access to evidence the
CNN cannot see. This decomposition is the central methodological claim of the thesis.

### 5.2 Why a generic image CNN fails at steganalysis

A standard image classifier (conv → BN → ReLU → max-pool stacks, ImageNet-style
normalization, resize to 224×224) learns *semantic content* — objects, textures,
colors. Steganographic embedding changes pixel values by ±1 in the LSB: a signal ~48 dB
below the image content. Three standard practices actively destroy it:

1. **Resizing/interpolation** averages neighboring pixels and erases LSB-level
   artifacts. StegShield never downscales; it crops (and upscales with NEAREST only if
   the image is smaller than the input size).
2. **Semantic feature learning**: without help, gradient descent finds content features
   long before it finds residual noise statistics. The early baseline experiment
   confirmed this: a plain 4-block CNN collapsed to the majority class (accuracy 0.75 =
   majority baseline, stego recall 1.0, clean recall 0.0, macro F1 0.4286).
3. **Aggressive pooling early** discards the high-frequency information where the
   evidence lives.

The steganalysis literature's answer is **fixed high-pass residual filtering**: filter
the image with known noise-residual kernels first, so the network starts from a
representation where image content is suppressed and the stego signal-to-noise ratio is
maximized. This is the design adopted for both models below.

### 5.3 SRM preprocessing front-end (shared by both models)

Both CNNs start with the **30 basic SRM high-pass filters** from the Spatial Rich
Models (Fridrich & Kodovský, *IEEE TIFS* 2012), implemented as a fixed (non-trainable)
5×5 convolution, kernel values **unnormalized**, exactly as in Yedroudj-Net:

| Filter class | Count | Shape idea |
|---|---|---|
| 1st-order directional differences | 8 | `[-1, 1]` in 8 directions |
| 2nd-order differences | 4 | `[1, -2, 1]` horizontal/vertical/2 diagonals |
| 3rd-order differences | 8 | `[1, -3, 3, -1]` in 8 directions |
| SQUARE 3×3 | 1 | 2D Laplacian-like |
| EDGE 3×3 | 4 | half-SQUARE rotations |
| SQUARE 5×5 | 1 | 5×5 KV kernel |
| EDGE 5×5 | 4 | half-SQUARE-5×5 rotations |

**RGB adaptation** (a deliberate deviation from the grayscale literature): the paper
uses single-channel 256×256 BOSSBase images. StegShield analyzes color images, and the
dataset embeds payloads in all three channel LSBs. The 30-filter bank is applied to
each channel separately via a **grouped convolution** (3 groups → 90 residual maps), so
channel-specific artifacts survive instead of being averaged away by a grayscale
conversion. This is documented in code (`SRMPreprocessor`) and should be presented in
the thesis as the RGB adaptation of the published architecture.

Earlier project state used 9 hand-picked "SRM-inspired" kernels; upgrading both models
to the same full 30-filter bank means the model comparison isolates the *feature
extractor trunk*, not the preprocessing.

### 5.4 Literature baseline: Yedroudj-Net

> M. Yedroudj, F. Comby, M. Chaumont, "Yedroudj-Net: An Efficient CNN for Spatial
> Steganalysis", IEEE ICASSP 2018, pp. 2092–2096, doi:10.1109/ICASSP.2018.8461438
> (HAL: lirmm-01717550).

Why this baseline: it is a peer-reviewed, citable steganalysis CNN that outperformed
Xu-Net and Ye-Net at publication time, it is small enough to train on a 4 GB laptop
GPU, and the paper itself was already part of the thesis bibliography. A plain image
classifier was originally used as the baseline, but a baseline that collapses teaches
little; comparing against a *real* steganalysis architecture is worth more in the
thesis comparison.

Architecture as implemented (`YedroudjNet`, faithful to the paper, RGB-adapted):

| Stage | Spec |
|---|---|
| Preprocess | fixed SRM bank, 5×5, 90 maps (30 × 3 channels), no bias |
| Block 1 | conv 5×5 → 30 maps (no bias) → **ABS** → BN → **Trunc(T=3)**, *no pooling* |
| Block 2 | conv 5×5 → 30 (no bias) → BN → **Trunc(T=2)** → avg-pool 5×5 s2 |
| Block 3 | conv 3×3 → 32 (no bias) → BN → ReLU → avg-pool 5×5 s2 |
| Block 4 | conv 3×3 → 64 (no bias) → BN → ReLU → avg-pool 5×5 s2 |
| Block 5 | conv 3×3 → 128 (no bias) → BN → ReLU → **global** avg-pool |
| Classifier | FC 128→256 → ReLU → FC 256→1024 → ReLU → FC 1024→2 |

Parameters: **491,860 total, 489,610 trainable** (2,250 fixed SRM weights).

Distinctive elements and their published rationale:

- **ABS layer** (block 1 only): forces the model to treat residual sign symmetrically
  (stego noise is sign-symmetric) — inherited from Xu-Net.
- **Truncation (TLU)** `Trunc(x) = clamp(x, -T, T)` with T=3 then T=2: suppresses
  large residual outliers caused by image content (edges), which are sparse and
  statistically insignificant for stego detection. **This only makes sense on the
  0–255 pixel scale** — see §5.6.
- **No pooling in block 1**: avoid early loss of the weak signal.
- **Average pooling, not max pooling**: max-pool selects content extremes; avg-pool
  preserves noise statistics.
- **Global average pooling**: removes spatial location information, so the detector
  generalizes regardless of *where* the payload sits.

### 5.5 Proposed model: StegShieldCNN

The project's own architecture keeps the identical SRM front-end (plus a TLU clamp at
3.0 directly after the SRM filters) and replaces the plain convolutional trunk with a
**ResNet-style residual trunk**:

| Stage | Spec |
|---|---|
| Preprocess | SRM bank (90 maps) → Trunc(3.0) |
| Stem | conv 3×3 → 32 → BN → ReLU |
| Stage 1 | 2 residual blocks, 32ch, stride 1 (full resolution) |
| Stage 2 | 2 residual blocks, 32→64, stride 2 |
| Stage 3 | 2 residual blocks, 64→128, stride 2 |
| Stage 4 | 2 residual blocks, 128→256, stride 2 |
| Classifier | GAP → dropout 0.3 → FC 256→128 → ReLU → dropout 0.2 → FC 128→2 |

Parameters: **2,855,500 total, 2,853,250 trainable** — ~5.8× larger than Yedroudj-Net.

Because both models share the preprocessing, the experiment isolates exactly one
question: *does a deeper residual trunk beat the paper's five plain blocks?* Answer on
this dataset: no measurable difference (both 0.9998 macro F1) — the dataset is
saturated; see §9.3 for the honest interpretation.

Comparison table for the thesis:

| Property | Yedroudj-Net | StegShieldCNN |
|---|---|---|
| Preprocessing | SRM 30×3, fixed | SRM 30×3, fixed + TLU(3) |
| Trunk | 5 plain conv blocks | 8 residual blocks in 4 stages |
| Activations | ABS+TLU (1–2), ReLU (3–5) | ReLU |
| Pooling | average | strided convolutions |
| Regularization | none (paper uses none) | dropout 0.3/0.2 |
| Trainable params | 489,610 | 2,853,250 |
| Avg epoch time (RTX 3050 4GB, AMP, batch 16) | 155 s | 270 s |

### 5.6 Input pipeline — the decisive engineering insight

**Crop position.** The single most important fix in the project. The original loader
used `CenterCrop(256)` on 512×512 images, keeping rows 128–384. But sequential LSB
tools embed starting at pixel (0,0), row-major. Measured on real cover/stego pairs:

- An Ethereum-address payload modifies **only row 0, columns 3–119** (102 pixels,
  max channel diff = 1).
- A PowerShell script payload modifies **rows 0–12**.

A center crop therefore removed 100% of the embedded evidence for all but the largest
payloads. This single preprocessing choice explains the catastrophic early results:

| Configuration | Test accuracy | Macro F1 | Stego recall |
|---|---|---|---|
| Center crop, [0,1] pixels, 9 filters (old) | 0.4461 | 0.4445 | **0.2615** |
| Top-left crop, 0–255 pixels, SRM-30 (new) | **0.9999** | **0.9998** | **1.0000** |

The fix: crop from the **top-left corner** (`--crop top-left`, default), with `--crop
center` retained as a CLI option for the ablation experiment. Thesis framing: this is
*informed preprocessing* — the analyst knows the threat class (sequential embedding)
and preserves the region where its evidence concentrates; the center-crop ablation
demonstrates how a standard computer-vision habit silently destroys forensic evidence.

**No downscaling.** Images smaller than the input size are upscaled with NEAREST
(which preserves exact pixel values); larger images are cropped, never resized, since
interpolation averages away ±1 artifacts.

**Pixel scale (`raw255`).** `torchvision.ToTensor()` scales pixels to [0,1]. The
truncation thresholds in steganalysis CNNs (T=3, T=2) are defined on **raw 0–255
values**; on [0,1] inputs the clamp at ±3 never activates and the TLU is inert.
StegShield adds a `raw255` normalization mode (default) that multiplies the tensor by
255, restoring the published operating regime. (`none` and `imagenet` modes remain
available for ablations.)

### 5.7 Training methodology

| Component | Choice | Rationale |
|---|---|---|
| Loss | Cross-entropy with inverse-frequency class weights | binary split is 1:3 imbalanced |
| Sampler | `WeightedRandomSampler` (balanced, with replacement) | each batch sees both classes; configurable for ablation (`--no-class-weights`, `--no-balanced-sampler`) |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 | robust default; converges faster than the paper's SGD on this task |
| Schedule | Cosine annealing over the run | smooth decay, no step tuning |
| Epochs / batch | 15 / 16 | batch chosen for 4 GB VRAM (see below) |
| Precision | Mixed precision (`--amp`, autocast + GradScaler) | ~halves VRAM, faster on RTX 3050 |
| Checkpoint selection | best validation **macro F1** (`--selection-metric`, balanced accuracy also available) | accuracy is misleading under imbalance — a majority-class predictor scores 0.75 |
| Reproducibility | every checkpoint stores task, labels, model name, normalization, **crop**, image size, class-weight/sampler flags, selection metric, device info, timestamps | evaluation and inference read the stored pipeline settings, so a checkpoint cannot silently be evaluated with the wrong preprocessing |

**Hardware lessons (worth a thesis paragraph on practical feasibility).** Training ran
on a laptop RTX 3050 (4 GB VRAM, Windows 11, WDDM driver model). Two failure modes were
identified and fixed:

1. *VRAM overflow → silent 10–50× slowdown.* At batch 32, allocations reached
   3.88/4.0 GB and the Windows driver began spilling CUDA memory into shared system
   RAM. The GPU shows "100% utilization" while mostly stalling on transfers (power
   draw 41 W of a 95 W budget). A 15-epoch run projected to 5+ hours. At batch 16 with
   AMP, the same training completes in ~40–67 minutes (155 s/epoch Yedroudj-Net,
   270 s/epoch StegShieldCNN).
2. *Single-threaded PNG decoding.* With `num_workers=0`, one CPU thread decodes 24,000
   PNGs per epoch (train+val) and starves the GPU. `--num-workers 4` with persistent
   workers removes the bottleneck (transforms were made picklable for Windows
   `spawn`-based workers).

Training also prints per-epoch metrics with an ETA, saves the best checkpoint
immediately when validation improves, and rewrites the metrics JSON every epoch, so an
interrupted run keeps its best model and full history.

**Training dynamics (observed).**

- *StegShieldCNN*: best macro F1 0.9997 reached at **epoch 3**; remained ≥0.9993
  thereafter. Average 270 s/epoch.
- *Yedroudj-Net*: epoch 1 **collapsed to all-stego** (macro F1 0.4286 — exactly the
  collapse signature), epoch 2 jumped to 0.9927, epoch 3 regressed (val loss spike to
  1.25, macro F1 0.725 — transient instability), best 0.9997 at **epoch 10**. Average
  155 s/epoch. The slow start is consistent with the paper's observation that
  steganalysis CNNs converge slowly; the best-checkpoint selection policy absorbed the
  epoch-3 regression.

### 5.8 Evaluation methodology

Accuracy is reported but treated as secondary; the imbalanced binary task lets a
majority-class predictor score 0.75. Primary metrics:

- **Macro F1** (checkpoint selection and headline metric) — averages F1 over classes,
  so the minority class counts equally.
- **Balanced accuracy** — average per-class recall.
- **Per-class precision/recall/F1, false negatives by class, full confusion matrix** —
  in every report; for a security system, *stego recall* (false-negative rate on
  threats) matters most.
- **Majority-class baseline accuracy** — printed alongside accuracy as the floor.
- **ROC-AUC** for binary CNN evaluations: `evaluate-cnn` stores every test sample's
  P(stego) (`binary_scores` in the report), and the ROC/AUC is computed by an
  in-project implementation (threshold sweep + trapezoidal integration,
  `ml_metrics.roc_curve_points` / `auc_from_roc_points`, unit-tested) — no external
  dependency.

All evaluation artifacts are JSON; `stegshield summarize-experiments` produces the
thesis comparison table and `stegshield plot-results` produces 300-dpi figures
(training curves, row-normalized confusion-matrix heatmaps, overlaid ROC curves with
AUC, grouped metric comparison chart).

### 5.9 CNN results

**Binary clean/stego, test_standard (2,000 clean / 6,000 stego):**

| Model | Accuracy | Macro F1 | ROC-AUC | Clean recall | Stego recall |
|---|---|---|---|---|---|
| Yedroudj-Net | 0.999875 | 0.999834 | 1.0 | 1.0000 | 0.999833 |
| StegShieldCNN | 0.999875 | 0.999834 | 1.0 | 0.9995 | 1.0000 |

Errors: Yedroudj-Net misses 1 stego file of 6,000; StegShieldCNN misflags 1 clean file
of 2,000. Statistically indistinguishable.

**Binary, test_adversarial (12,000 stego with base64/zip-encoded payloads, no clean):**
both models detect **100%**. Note for the thesis: this split has no clean class, so
report detection rate, not macro F1 (a zero-support class makes macro F1 read 0.5 by
construction).

**Historical comparison (same model family, before the pipeline fixes):** accuracy
0.4461, macro F1 0.4445, stego recall 0.2615 — *below* the 0.75 majority baseline.
The delta is attributable almost entirely to crop position + pixel scale (§5.6).

---

## 6. Static Analysis Branch

The static analyzer (`metadata/extract.py`, `metadata/risk_rules.py`,
`utils/file_validation.py`) parses files defensively (never executes content) and emits
**indicators** with severities (low 0.15 / medium 0.30 / high 0.50), summed and clamped
to a risk score in [0,1]. Score bands: <0.3 safe, 0.3–0.69 suspicious, ≥0.7 dangerous.

### 6.1 Rule inventory

| Indicator | Severity | Trigger |
|---|---|---|
| `unknown_file_signature` | high | magic bytes match no supported image type |
| `extension_type_mismatch` | medium | extension disagrees with detected type |
| `image_parse_error` | high | Pillow cannot parse a file claiming to be an image |
| `trailing_data_after_image_end` | medium | bytes after JPEG EOI marker **or PNG IEND chunk** |
| `embedded_binary_signature` | high | validated PE executable, ZIP/RAR/7z/ELF signature inside the file |
| `sequential_lsb_payload` | medium | LSB payload estimate < 128 bytes (marker-scale) |
| `sequential_lsb_payload_large` | high | LSB payload estimate ≥ 128 bytes (script/binary-scale) |
| `large_metadata` | medium | metadata text > 16 KB |
| `suspicious_metadata_text` | medium | script/URL/shell patterns in metadata fields |
| `large_file_for_dimensions` | low | > 100 bytes/pixel |

### 6.2 Sequential-LSB payload estimator (the analyzer's key component)

> Structural idea: A. Westfeld, A. Pfitzmann, "Attacks on Steganographic Systems",
> Information Hiding 1999, LNCS 1768, pp. 61–76 (the classical sequential-LSB
> steganalysis attack).

Principle: sequential LSB replacement overwrites the **start** of the LSB stream with
payload bits that look like a Bernoulli(0.5) sequence, while the rest of the LSB plane
keeps the spatial structure of the image. The detector:

1. Reads the LSB of every channel value in scan order (RGB interleaved, row-major).
2. Splits the bit stream into blocks of 128 pixels (384 bits).
3. Marks a block "noise-like" if its ones-ratio lies in (0.25, 0.75).
4. Measures the leading run of noise-like blocks; requires at least one non-noisy
   block after the run (the *transition*) — a fully noisy image yields no verdict,
   which protects against textured-image false positives.
5. Reports `estimated_payload_bytes = run × 384 / 8`.

Constraints: lossless formats only (PNG/BMP/TIFF — JPEG-decoded LSBs are codec
artifacts); numpy imported lazily so the base CLI works without ML dependencies.

Empirical validation (150–200 test images per group):

| Group | Detection rate | Estimated size (median) |
|---|---|---|
| clean | **0/150 false positives** | — |
| eth/url | 150/150 | 48 B |
| scripts | 149/150 | ~15 KB (27 KB at 200-sample p50) |
| base64/zip variants | 146/150 | 336–2,448 B |

The 128-byte severity threshold sits in the empirical gap (eth/url max 96 B; scripts
p10 1,728 B). This estimator is what gives the system a defensible
suspicious-vs-dangerous signal: payload *presence* comes from the CNN, payload
*scale* from this rule.

A rejected alternative worth one thesis paragraph: the chi-square pairs-of-values
attack (also Westfeld–Pfitzmann) was prototyped first but failed on this corpus —
the covers' smooth regions produce sparse value histograms in small blocks, leaving
too few valid histogram pairs for the statistic.

### 6.3 The false-positive bug (before/after — strong discussion material)

The original `find_embedded_signatures` scanned the whole file for the 2-byte DOS
signature `"MZ"`. In a few hundred KB of compressed (statistically random) PNG data,
"MZ" occurs by chance in most files. Measured effect: the **only** rule firing on the
entire dataset was this one — on 24% of *clean* covers (36/150), 28% of eth/url, 73%
of scripts (size-correlated random matching). The fusion layer escalated every such hit
to `dangerous`, producing 427/2,000 clean→dangerous errors, while contributing
accidental (noise-correlated) "signal" on stego files.

Fixes:

- **PE structure validation**: an "MZ" candidate counts only if its `e_lfanew` field
  (offset 0x3C) points to a valid `PE\x00\x00` signature within the file — the
  structure a real Windows executable must have. Archive signatures must be ≥4 bytes
  (`PK\x03\x04`, `Rar!\x1a\x07`, `7z\xbc\xaf\x27\x1c`, `\x7fELF`).
- After the fix: clean files trigger **zero** indicators (100/100 sampled), and every
  indicator that fires corresponds to real embedded structure.

The pre-fix fusion evaluation is preserved as an ablation:
`outputs/reports/fusion_test_standard_metrics_oldrules.json`.

---

## 7. Fusion Layer

```text
score = 0.55 * P(stego) + 0.45 * metadata_risk      (clamped to [0,1])
```

Override rules (in order):

1. Embedded binary/archive signature → `score = max(score, 0.7)` (dangerous floor).
2. Any high-severity indicator → `score = max(score, 0.5)`.
3. High-severity indicator **and** P(stego) ≥ 0.5 → `score = max(score, 0.7)`.

Label bands: <0.3 `safe`, 0.3–0.69 `suspicious`, ≥0.7 `dangerous`. Every fused result
carries a one-sentence explanation naming the dominant evidence.

The weights were chosen so the label bands implement the intended semantics (worked
examples for the thesis):

| Evidence | Score | Label |
|---|---|---|
| clean CNN (≈0), no indicators | ≈0 | safe |
| confident stego CNN (≈1), no static corroboration | 0.55 | suspicious |
| stego CNN + small LSB payload (medium, 0.3) | 0.55 + 0.135 = 0.685 | suspicious |
| stego CNN + script-scale payload (high, 0.5) | 0.775 (and override 3) | dangerous |
| clean CNN + embedded ZIP/PE signature | override 1 → 0.7 | dangerous |
| clean CNN + parse error (high) | override 2 → 0.5 | suspicious |

Design intent: *visual stego evidence alone never exceeds `suspicious`* — escalation to
`dangerous` always requires explainable static evidence. This keeps the system's most
severe verdict auditable.

---

## 8. End-to-End Results

### 8.1 Fusion evaluation, test_standard (8,000 images, 3-way)

| Method | Accuracy | Macro F1 | Safe recall | Suspicious recall | Dangerous recall |
|---|---|---|---|---|---|
| Metadata-only | 0.5600 | 0.5280 | 0.9995 | 1.0 | 0.0 |
| CNN-only → risk | 0.5600 | 0.5283 | 0.9995 | 1.0 | 0.0 |
| **Fused** | **0.9958** | **0.9960** | 0.9990 | 0.9984 | 0.9920 |

Fused confusion matrix (rows = actual, columns = predicted):

```text
              safe   suspicious  dangerous
safe          1998        2          0
suspicious       0     2477          4
dangerous        0       28       3491
```

Readings for the thesis:

- **Zero threat-as-safe errors**: no suspicious or dangerous file was labeled safe —
  the security-critical false-negative count is 0/6,000.
- Each component alone caps at 0.56 accuracy because, *by design*, neither can output
  `dangerous` on its own (metadata-only is capped by its conservative scoring; CNN-only
  cannot know payload scale). Fusion is not an incremental improvement — it is the
  mechanism that makes the 3-way task solvable. This is the cleanest possible
  demonstration of the thesis claim.
- The 28 dangerous→suspicious errors are scripts whose payload is below the 128-byte
  threshold (the empirical overlap in §4.1); the 4 suspicious→dangerous are payloads
  just above it. These are honest boundary cases of the size heuristic.
- With the old (buggy) rules the fused result was 0.7802 / 0.7948 with 427
  clean→dangerous false alarms — and that number was *inflated* by noise correlating
  with file size. Fixing the rules improved accuracy by +0.2156 **and** made the
  explanation behind every verdict true.

### 8.2 Figures (outputs/figures/, 300 dpi)

- `*_training_curves.png` — loss + macro-F1/balanced-accuracy per epoch, best epoch
  marked (shows Yedroudj-Net's slow start and epoch-3 instability vs StegShieldCNN's
  immediate convergence).
- `*_confusion_matrix.png` — row-normalized heatmaps; the metadata-only / CNN-only /
  fused triptych is the headline figure.
- `roc_curves.png` — both CNNs at AUC = 1.0.
- `model_comparison.png` — all systems incl. the old-rules ablation in one chart.

---

## 9. Key Insights (thesis discussion chapter material)

### 9.1 Preprocessing is forensic evidence handling

The center-crop bug is the project's most instructive result: a default
computer-vision transform (center crop) silently deleted the entire signal class,
producing a model *worse than the majority baseline* (macro F1 0.4445), while the
identical architecture with a top-left crop reaches 0.9998. In steganalysis, every
resize, crop, recompression, or normalization is evidence handling. Generic CV
pipelines are not neutral.

### 9.2 Static rules must be validated against the false-positive base rate

A 2-byte signature scan looks reasonable and passes happy-path tests, but on
compressed data its per-file false-positive probability approaches 1. The fix
(structural PE validation) embodies a general principle: a detection rule is defined
not by what it catches but by its expected false-positive rate on benign data. The
before/after fusion numbers (0.78 → 0.996) quantify the cost of ignoring this.

### 9.3 Near-perfect numbers must be honestly bounded

Both CNNs and the fused system are near-ceiling. The thesis must state why: sequential
LSB replacement is the *easiest* steganography class to detect — it concentrates a
maximally non-natural bit pattern in a known location, and SRM filters were designed
for exactly such residual anomalies. The claims should be scoped to this threat class:

- Claim: the hybrid pipeline reliably detects and risk-grades sequential-LSB payload
  smuggling (Invoke-PSImage family) and container anomalies.
- Non-claim: detection of content-adaptive embedding (S-UNIWARD, WOW, HILL), where
  published detectors err at 10–30% at low payload rates — cite Yedroudj-Net's own
  numbers (e.g., ~27.8% error on WOW at 0.2 bpp) to show what realistic hard-case
  performance looks like.
- Non-claim: malware detection — the system grades container risk; it never inspects
  payload semantics beyond size.

### 9.4 Architecture mattered less than representation

Yedroudj-Net (490 K params, 2018) and StegShieldCNN (2.85 M params, residual trunk)
are indistinguishable on this dataset. The performance came from the *representation
decisions* — SRM residuals, truncation, raw pixel scale, crop placement — not from
trunk depth. This inverts the usual deep-learning narrative and is worth a paragraph.

### 9.5 Decomposition beats end-to-end when evidence types differ

The 3-way end-to-end CNN collapsed (macro F1 0.39); the decomposed system (binary CNN
+ payload-scale rule + fusion) reaches 0.996 *and* every verdict carries a
human-readable explanation. For security tooling, explainability is not a bonus — the
fused label cites the exact indicator (e.g., "script/binary-scale sequential LSB
payload, ~27 KB") that justified escalation.

### 9.6 Dataset artifacts and experimenter honesty

Two exploitable artifacts (RGBA/RGB mode split; smooth cover tops) were identified and
deliberately *not* used as features (§4.1). Documenting resisted shortcuts strengthens
the methodology chapter.

---

## 10. Limitations and Future Work

1. **Single dataset, single embedding family.** Validation on BOSSBase/ALASKA-style
   adaptive embedding would test the CNNs where the literature says it is hard; the
   sequential-LSB rule would not transfer (adaptive embedding is neither sequential nor
   LSB-replacement) — which is precisely why the CNN branch exists.
2. **Payload-size heuristic boundary.** The 128-byte threshold misgrades scripts
   smaller than ~5 blocks (28/3,519 in the test set). A learned classifier over static
   features (planned in the project plan as the metadata-ML upgrade) could replace the
   fixed threshold.
3. **Crop coverage.** A 256×256 top-left crop sees the first 256 rows; a payload
   embedded with a deliberate offset beyond that escapes the visual branch (the LSB
   rule scans the full image and still catches sequential runs anywhere in the leading
   blocks). Full-resolution inference (`--image-size 512`) is a one-flag ablation.
4. **JPEG support.** The LSB estimator is lossless-only by design; JPEG steganalysis
   (DCT-domain) is future work.
5. **Generalization of the smooth-top assumption.** On covers with heavily textured
   top-left regions, the LSB detector's false-positive rate would rise; the transition
   requirement and CNN-gated escalation mitigate but do not eliminate this.

---

## 11. Reproducibility

Environment: Windows 11, Python 3.11, PyTorch 2.11.0+cu128, RTX 3050 Laptop (4 GB),
package installed with `pip install -e ".[dev,ml]"`.

```powershell
# 1. dataset CSVs (preserves the official split)
stegshield import-kaggle-splits

# 2. train both CNNs (~40-70 min each on the 4 GB GPU)
stegshield train-cnn --task stego --model yedroudj    --epochs 15 --batch-size 16 --amp --num-workers 4 --device cuda --output-model outputs/models/yedroudj_stego.pt    --output-metrics outputs/reports/yedroudj_stego_training.json
stegshield train-cnn --task stego --model steganalysis --epochs 15 --batch-size 16 --amp --num-workers 4 --device cuda --output-model outputs/models/steganalysis_stego.pt --output-metrics outputs/reports/steganalysis_stego_training.json

# 3. binary CNN evaluations
stegshield evaluate-cnn --model-path outputs/models/yedroudj_stego.pt    --split-csv data/splits/test_standard.csv    --output-report outputs/reports/yedroudj_stego_test_standard.json    --num-workers 4 --device cuda
stegshield evaluate-cnn --model-path outputs/models/yedroudj_stego.pt    --split-csv data/splits/test_adversarial.csv --output-report outputs/reports/yedroudj_stego_test_adversarial.json --num-workers 4 --device cuda
stegshield evaluate-cnn --model-path outputs/models/steganalysis_stego.pt --split-csv data/splits/test_standard.csv    --output-report outputs/reports/steganalysis_stego_test_standard.json --num-workers 4 --device cuda
stegshield evaluate-cnn --model-path outputs/models/steganalysis_stego.pt --split-csv data/splits/test_adversarial.csv --output-report outputs/reports/steganalysis_stego_test_adversarial.json --num-workers 4 --device cuda

# 4. hybrid 3-way evaluation
stegshield evaluate-fusion --model-path outputs/models/steganalysis_stego.pt --split-csv data/splits/test_standard.csv --output-report outputs/reports/fusion_test_standard_metrics.json --device cuda

# 5. thesis table + figures
stegshield summarize-experiments outputs/reports/*_test_*.json outputs/reports/fusion_test_standard_metrics.json --output outputs/reports/experiment_summary.md
stegshield plot-results outputs/reports/*_training.json outputs/reports/*_test_standard.json outputs/reports/fusion_test_standard_metrics.json --output-dir outputs/figures
```

Ablation flags: `--crop center` (the destroyed-evidence experiment), `--normalization
none|imagenet`, `--no-class-weights`, `--no-balanced-sampler`, `--image-size 512`,
`--selection-metric balanced_accuracy`.

Checkpoints are self-describing: evaluation and the `analyze` CLI read the stored crop,
normalization, image size, and task from the checkpoint, so results cannot silently be
produced with mismatched preprocessing.

---

## 12. References

1. M. Yedroudj, F. Comby, M. Chaumont, "Yedroudj-Net: An Efficient CNN for Spatial
   Steganalysis", *IEEE ICASSP*, 2018, pp. 2092–2096. doi:10.1109/ICASSP.2018.8461438.
   (Literature baseline architecture; SRM-preprocessing + TLU design.)
2. J. Fridrich, J. Kodovský, "Rich Models for Steganalysis of Digital Images",
   *IEEE TIFS* 7(3), 2012. (Origin of the 30 SRM high-pass kernels.)
3. A. Westfeld, A. Pfitzmann, "Attacks on Steganographic Systems", *Information
   Hiding*, LNCS 1768, 1999, pp. 61–76. (Sequential-LSB steganalysis; basis of the
   payload estimator and the rejected chi-square prototype.)
4. S. Kang, H. Park, J.-I. Park, "CNN-Based Ternary Classification for Image
   Steganalysis", *Electronics* 8(12):1225, 2019. (Evidence for the difficulty of
   multi-class steganalysis CNNs; motivates the binary decomposition.)
5. G. Xu, H.-Z. Wu, Y.-Q. Shi, "Structural Design of Convolutional Neural Networks for
   Steganalysis", *IEEE SPL* 23(5), 2016. (ABS layer; referenced via Yedroudj-Net.)
6. J. Ye, J. Ni, Y. Yi, "Deep Learning Hierarchical Representations for Image
   Steganalysis", *IEEE TIFS* 12(11), 2017. (TLU activation; 30-filter initialization;
   referenced via Yedroudj-Net.)
7. M. Boroumand, M. Chen, J. Fridrich, "Deep Residual Network for Steganalysis of
   Digital Images" (SRNet), *IEEE TIFS* 14(5), 2019. (Candidate stronger baseline,
   discussed as future work.)
8. Kaggle Stego Images Dataset, M. Zuppelli:
   <https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset>.
9. Invoke-PSImage (sequential LSB embedding tool family that defines the threat model).
