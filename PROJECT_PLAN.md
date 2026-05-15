# Bachelor Thesis Project Plan

## Project Title

StegShield: AI-Based Detection of Suspicious and Dangerous Image Files Using CNN Visual Analysis and File Metadata Inspection

## 1. Project Goal

The goal of this project is to build an AI-assisted system that analyzes image files and classifies them as `safe`, `suspicious`, or `dangerous`. The system will combine two complementary sources of evidence:

1. Visual content analysis using a Convolutional Neural Network (CNN).
2. File metadata and structural analysis using metadata extraction and security-focused file inspection.

The final system should accept an image file as input, extract visual and metadata-based features, process them through a classification pipeline, and output a risk label such as:

- `safe`
- `suspicious`
- `dangerous`

The main thesis idea is not only to classify images by what they visually show, but also to detect cases where an image file may carry suspicious payloads, manipulated metadata, hidden content, malformed structure, or other indicators of abuse.

## 2. Problem Background

Image files can be abused in multiple ways:

- They can contain malicious payloads hidden through steganography.
- They can include suspicious metadata, scripts, comments, GPS/location data, creator-tool traces, or malformed fields.
- They can be crafted to exploit vulnerabilities in image parsers, viewers, or metadata processors.
- They can be used as carriers in phishing, malware delivery, command-and-control workflows, or data exfiltration.

A pure CNN can learn visual patterns, but it may miss threats that are not visible in the pixel data. Metadata and structural inspection can detect suspicious file-level indicators, but it may miss visual or statistical signs of hidden payloads. Combining both approaches should provide a stronger detection pipeline.

## 3. Scope

### In Scope

- Image file ingestion.
- Image preprocessing for CNN input.
- CNN-based classification using pixel data.
- Metadata extraction from image files.
- Rule-based or ML-based analysis of metadata features.
- Fusion of CNN output and metadata risk indicators.
- An installable command-line interface for testing files locally.
- A later API mode that can be integrated into applications that accept uploaded images, such as profile pictures or user-generated content.
- Evaluation using accuracy, precision, recall, F1-score, confusion matrix, and false positive/false negative analysis.
- Documentation suitable for bachelor thesis chapters.

### Out of Scope for Initial Version

- Real-time antivirus replacement.
- Full malware execution or sandboxing.
- Reverse engineering binary payloads.
- Detecting every possible steganography method.
- Cloud deployment unless added later.
- Full production API hardening in the first version.
- Production-grade enterprise security integration.

## 4. Proposed System Architecture

The system will be built as a pipeline:

```text
Input image file
      |
      v
File validation and type detection
      |
      +-----------------------------+
      |                             |
      v                             v
Image preprocessing          Metadata extraction
      |                             |
      v                             v
CNN visual classifier        Metadata feature analyzer
      |                             |
      +-------------+---------------+
                    |
                    v
          Fusion / decision layer
                    |
                    v
        Final label + explanation
```

## 5. Core Components

### 5.1 Image Input Module

Responsibilities:

- Accept image files such as `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, and `.webp`.
- Verify that the file is actually an image and not only renamed with an image extension.
- Record basic file properties:
  - file name
  - file extension
  - file size
  - MIME type
  - hash values such as SHA-256
  - image dimensions
  - color mode

Possible tools:

- Python `pathlib`
- `Pillow`
- `python-magic` or MIME detection
- `hashlib`

### 5.2 Image Preprocessing Module

Responsibilities:

- Load the image safely.
- Convert image to RGB.
- Resize image to the CNN input size, for example `224x224`.
- Normalize pixel values.
- Handle corrupted or unreadable files.
- Optionally generate additional statistical features:
  - color histograms
  - noise level
  - entropy
  - compression artifacts

Possible tools:

- `Pillow`
- `OpenCV`
- `torchvision.transforms`
- `numpy`

### 5.3 CNN Visual Classifier

The CNN will analyze image pixels and produce a probability score for each class.

Planned model approach:

1. Build a custom CNN from scratch as the main bachelor thesis model.
2. Compare it against at least one pretrained transfer-learning model, if the professor accepts this comparison.

The custom CNN is useful for the thesis because it demonstrates understanding of the architecture, training process, loss function, optimization, and evaluation. The transfer-learning model is useful as a benchmark because it shows how the custom model performs against a stronger existing architecture.

Recommended custom CNN baseline:

- Input: `224x224x3` RGB image.
- Convolution block 1: convolution, batch normalization, ReLU, max pooling.
- Convolution block 2: convolution, batch normalization, ReLU, max pooling.
- Convolution block 3: convolution, batch normalization, ReLU, max pooling.
- Convolution block 4: convolution, batch normalization, ReLU, global average pooling.
- Fully connected layer with dropout.
- Output layer with 3 classes.

Target classes:

- `safe`: normal image with no suspicious visual, metadata, or structural indicators.
- `suspicious`: image with weak or moderate risk indicators, such as unusual metadata, steganographic traits, high entropy, or minor structural anomalies.
- `dangerous`: image with strong indicators, such as known stego sample labeling, executable/archive signatures, payload-like appended data, severe format mismatch, or multiple high-risk signals.

Possible comparison model:

- ResNet-18
- EfficientNet-B0
- MobileNetV3

The first implementation will start with the custom CNN. Transfer learning will be added as a comparison experiment after the baseline pipeline is stable.

### 5.4 Metadata Extraction Module

This module will inspect the image file without relying only on pixels.

Metadata to extract:

- EXIF fields
- camera/device data
- software/tool used to create or edit the file
- GPS metadata
- timestamps
- comments
- ICC profiles
- XMP metadata
- embedded thumbnails
- unusual metadata length
- unknown or suspicious metadata tags
- file size compared to image dimensions
- mismatch between extension and actual file signature
- trailing data after the expected image end marker
- unusually high entropy regions

Possible tools:

- `exiftool`
- `Pillow`
- `piexif`
- `hachoir`
- `python-magic`
- custom binary inspection for magic bytes and trailing data

Important note:

The metadata module should not execute any embedded content. It should only parse and inspect files defensively.

### 5.5 Metadata Risk Analyzer

The metadata analyzer will convert extracted metadata into risk signals.

Example risk indicators:

- File extension does not match detected MIME type.
- Very large file size for small image dimensions.
- Suspicious comments or metadata strings.
- Metadata contains script-like content.
- Metadata contains URLs, shell commands, base64-like blobs, or executable signatures.
- EXIF tool field mentions suspicious software.
- File contains data after JPEG end marker.
- File contains embedded archive signatures such as ZIP, RAR, or PE headers.
- Image fails normal parsing but still has an image extension.
- Entropy is unusually high.

Possible implementation approaches:

1. Rule-based scoring.
2. Classical ML model using metadata features.
3. Hybrid approach: rules first, then train a model once enough labeled samples exist.

Recommended first version:

- Rule-based metadata risk score from `0.0` to `1.0`.
- Later, convert metadata features into a machine learning classifier.

### 5.6 Fusion / Decision Layer

The decision layer combines:

- CNN probability score.
- Metadata risk score.
- Optional structural anomaly score.

Example formula:

```text
final_risk = 0.55 * cnn_risk + 0.45 * metadata_risk
```

Example labels:

```text
0.00 - 0.39 -> safe
0.40 - 0.69 -> suspicious
0.70 - 1.00 -> dangerous
```

The weighting should be tuned during evaluation. If metadata indicators are very strong, they may override the CNN result. For the three-label setup, the CNN can output three probabilities, while the metadata analyzer outputs a risk score and rule-trigger list. The fusion layer should keep the final decision explainable.

Example override rules:

- If the file contains executable headers after image data, label at least `suspicious`.
- If the file extension and MIME type are inconsistent, label at least `suspicious`.
- If parsing fails but file claims to be an image, label at least `suspicious`.
- If multiple high-risk indicators are triggered, label as `dangerous` even if the CNN prediction is uncertain.

## 6. Dataset Strategy

The dataset is the most important part of this project. The model will only be as strong as the data used to train and evaluate it.

### 6.1 Safe Image Dataset

Possible safe image sources:

- COCO dataset
- ImageNet subset
- Open Images dataset
- personal clean image collection
- public royalty-free images

Safe samples should include multiple file formats and image types:

- photos
- screenshots
- generated images
- edited images
- compressed images
- images with normal EXIF metadata
- images without metadata

### 6.2 Dangerous or Suspicious Image Dataset

This part needs careful design because collecting real malicious files can be risky and may not be allowed. The preferred direction is to use research steganography datasets first, because they provide real stego images without requiring executable malware handling.

Candidate datasets:

1. Kaggle Stego Images Dataset:
   - User-provided candidate: <https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset>
   - Useful as a practical starting point if the license, file structure, labels, and sample quality are acceptable.
   - Should be validated before becoming the main dataset.

2. StegoAppDB:
   - Source: <https://forensicstats.org/stegoappdb/>
   - License information: <https://forensicstats.org/stegoappdb-license/>
   - Strong research source because it contains innocent and stego mobile images, cover-stego pairs, and side information such as device, EXIF data, stego app, message information, and embedding rate.
   - Publicly available for scientific, non-commercial academic research, with citation and license conditions.
   - Full dataset is very large, so this project should download a selected subset only.

3. BOSSBase / BOWS2-based datasets:
   - Common benchmark sources in steganalysis research.
   - Useful for controlled experiments, especially where cover-stego pairs are available.
   - May require additional scripts to generate stego variants depending on the source.

4. ALASKA steganalysis datasets:
   - Useful for JPEG steganalysis benchmarking.
   - More complex but valuable for comparison if time allows.

5. JPEG StegoChecker dataset:
   - Source: <https://www.kaggle.com/datasets/h2020simargl/jpeg-stegochecker-dataset>
   - Based on BOSSBase and includes features for clean and stego JPEG experiments.
   - Useful as an additional comparison dataset, but we need to verify whether it provides images or extracted features only before using it for CNN training.

Synthetic suspicious samples are still useful for metadata and structural testing:

1. Generate synthetic suspicious samples:
   - images with abnormal metadata
   - images with long comments
   - images with appended random bytes
   - images with embedded harmless dummy payload markers
   - images with hidden text using steganography tools
   - images with mismatched extension and MIME type

2. Use security research datasets only if legally and safely accessible.

Important safety rule:

Do not execute real malware samples for this project. For the first thesis version, "dangerous" should mean dangerous or suspicious image container behavior, steganographic payload presence, or strong structural indicators, not live malware execution. If truly malicious files are ever introduced, they must be handled only with university approval, in an isolated virtual machine or lab environment.

### 6.3 Labeling Strategy

Initial labels:

- `safe`: normal image file with no suspicious indicators.
- `suspicious`: image file with unusual metadata, possible steganographic traits, weak structural anomalies, or uncertain model confidence.
- `dangerous`: image file with known stego labeling, simulated payload, suspicious embedded content, malformed structure, or strong multi-signal evidence.

Optional labels:

- `safe_clean`
- `safe_with_metadata`
- `suspicious_metadata`
- `stego_suspected`
- `malformed`
- `payload_appended`

Mapping rule for first dataset version:

- Clean cover/original images become `safe`.
- Stego images from known steganography datasets become at least `suspicious`.
- Stego images with high-confidence model output or additional metadata/structural indicators may be labeled `dangerous`.
- Synthetic files with appended archive/executable signatures, severe format mismatches, or payload-like trailing data become `dangerous`.

### 6.4 Dataset Split

Recommended split:

- 70% training
- 15% validation
- 15% testing

Important:

- Avoid near-duplicate images across train and test sets.
- Keep the `safe`, `suspicious`, and `dangerous` classes as balanced as practical.
- Keep a separate final test set that is not used during model tuning.

## 7. Model Training Plan

### 7.1 Baseline

Build simple baselines first:

- Metadata-only rule classifier.
- Custom CNN-only classifier.
- Optional pretrained CNN classifier for comparison.
- Combined CNN plus metadata classifier.

This will allow comparison and make the thesis stronger.

### 7.2 Training Steps

1. Collect and organize dataset.
2. Create labels file such as `labels.csv`.
3. Implement image preprocessing.
4. Train custom CNN model from scratch.
5. Validate on held-out validation set.
6. Tune hyperparameters.
7. Optionally train a transfer-learning model for comparison.
8. Extract metadata features for all samples.
9. Implement metadata risk score.
10. Combine CNN and metadata outputs.
11. Evaluate on final test set.

### 7.3 Metrics

Use the following evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1 for balanced three-class evaluation
- Per-class precision and recall
- ROC-AUC only for binary sub-experiments or one-vs-rest analysis
- Confusion matrix
- False positive rate
- False negative rate

For a security project, recall is especially important because false negatives mean dangerous files are missed. However, precision also matters because too many false alarms make the system less useful.

## 8. Suggested Technology Stack

Language:

- Python

Machine learning:

- PyTorch
- torchvision
- scikit-learn

Image processing:

- Pillow
- OpenCV
- numpy

Metadata and file inspection:

- exiftool
- piexif
- python-magic
- hachoir

Data handling:

- pandas
- numpy

Experiment tracking:

- CSV logs first
- optional later: MLflow, Weights & Biases, or TensorBoard

Interface:

- Installable CLI first
- FastAPI backend later
- Optional lightweight web UI only after the core model works

Working project/tool name:

- `stegshield`

Example final usage:

```text
stegshield analyze ./image.png
stegshield analyze ./image.png --json
stegshield batch ./uploads/ --output report.csv
```

The CLI-first approach is the most practical for the bachelor thesis because it is easy to demonstrate, easy to test, and directly supports local machine usage. The API can reuse the same internal analysis pipeline later.

## 9. Proposed Repository Structure

```text
Lic/
  PROJECT_PLAN.md
  README.md
  requirements.txt
  pyproject.toml
  data/
    raw/
      safe/
      suspicious/
      dangerous/
    processed/
    splits/
      train.csv
      val.csv
      test.csv
  notebooks/
    exploration.ipynb
  src/
    config.py
    dataset.py
    preprocess.py
    train_cnn.py
    evaluate.py
    predict.py
    cli.py
    api.py
    metadata/
      extract.py
      risk_rules.py
      features.py
    models/
      cnn.py
      pretrained.py
      fusion.py
    utils/
      hashing.py
      file_validation.py
  tests/
    test_metadata_rules.py
    test_file_validation.py
  outputs/
    models/
    reports/
    figures/
```

## 10. Milestones

### Milestone 1: Research and Definition

Deliverables:

- Final problem statement.
- Threat model.
- Dataset decision.
- Initial thesis bibliography.
- This project plan.

### Milestone 2: Dataset Preparation

Deliverables:

- Safe image dataset.
- Suspicious/dangerous image dataset.
- Validated subset from the Kaggle Stego Images Dataset or StegoAppDB.
- Dataset labeling file.
- Train/validation/test split.
- Basic dataset statistics.

### Milestone 3: Metadata Analyzer

Deliverables:

- Metadata extraction script.
- File signature validation.
- Risk rules.
- Metadata-only classifier baseline.
- Report showing common suspicious indicators.

### Milestone 4: CNN Model

Deliverables:

- Image dataset loader.
- Custom CNN training script.
- Validation metrics.
- Saved trained model.
- Custom CNN-only baseline results.
- Optional pretrained CNN comparison results.

### Milestone 5: Fusion Model

Deliverables:

- Combined decision layer.
- Final prediction script.
- Comparison between metadata-only, custom CNN-only, optional pretrained CNN, and combined system.
- Error analysis.

### Milestone 6: Interface and Demonstration

Deliverables:

- Installable CLI command named `stegshield`.
- File path input.
- Optional JSON output for API integration.
- Output label, confidence, metadata risk indicators, and explanation.
- Demo examples.

### Milestone 7: API Prototype

Deliverables:

- FastAPI wrapper around the same analysis pipeline.
- Endpoint for image upload.
- JSON response with final label, confidence, and triggered indicators.
- Basic validation for uploaded file size and type.

### Milestone 8: Thesis Writing

Deliverables:

- Introduction.
- Literature review.
- Methodology.
- Implementation.
- Experiments and results.
- Conclusion and future work.

## 11. Risks and Mitigations

### Risk: Not enough dangerous samples

Mitigation:

- Use synthetic suspicious files.
- Use public steganography datasets.
- Clearly explain the dataset limitations in the thesis.

### Risk: CNN may not detect hidden payloads well

Mitigation:

- Treat CNN as one signal, not the only detector.
- Use metadata and structural inspection as a major part of the system.
- Compare custom CNN-only against metadata-only, optional pretrained CNN, and combined results.

### Risk: Metadata can be missing or removed

Mitigation:

- Do not rely only on metadata.
- Include file structure, entropy, and pixel-based features.

### Risk: Real malware handling may be unsafe

Mitigation:

- Avoid executing any suspicious file.
- Prefer steganography datasets over live malware datasets.
- Use harmless synthetic payload markers for structural tests.
- Work in a virtual machine if risky files are ever introduced.
- Get explicit professor/university approval before using truly malicious samples.

### Risk: False positives

Mitigation:

- Evaluate against diverse safe images.
- Separate suspicious from dangerous when confidence is uncertain.
- Provide explanations for each decision.

## 12. Thesis Contribution

The thesis contribution can be framed as:

- A hybrid detection pipeline for suspicious image files.
- A three-label risk classification approach: `safe`, `suspicious`, and `dangerous`.
- A comparison of custom CNN-only, metadata-only, optional pretrained CNN, and combined detection methods.
- A practical evaluation of how metadata and structural file indicators improve image threat classification.
- A safe experimental framework using public steganography datasets and synthetic suspicious samples.

## 13. First Implementation Plan

The first working version should be simple and reliable.

### Version 0.1

Features:

- Create installable Python package skeleton.
- Add CLI command: `stegshield analyze <image_path>`.
- Validate file type.
- Extract basic metadata.
- Calculate hash and file size.
- Detect extension/MIME mismatch.
- Detect unusually large metadata.
- Detect suspicious strings in metadata.
- Output a metadata risk score.

### Version 0.2

Features:

- Add dataset folder structure.
- Add CSV labels.
- Add image preprocessing.
- Train a custom CNN from scratch.
- Save the trained model.

### Version 0.3

Features:

- Add CNN inference.
- Combine CNN score with metadata score.
- Output final label and explanation.

### Version 0.4

Features:

- Add evaluation scripts.
- Generate confusion matrix and metrics.
- Compare metadata-only, custom CNN-only, optional pretrained CNN, and fusion system.

### Version 0.5

Features:

- Polish installable CLI.
- Add JSON output mode.
- Prepare thesis figures and tables.
- Improve documentation.

### Version 0.6

Features:

- Add FastAPI endpoint for upload analysis.
- Reuse the same pipeline as the CLI.
- Add API response schema for integration into an upload-checking application.

## 14. Questions to Decide Next

These questions should be answered before implementation becomes too large:

1. Confirm with the professor whether transfer learning is acceptable as a comparison model.
2. Confirm with the professor whether real malicious files are allowed, or whether research stego datasets are enough.
3. Decide the final tool name. Current working name: `stegshield`.
4. Decide whether JPEG and PNG are enough for the first version.
5. Decide whether metadata analysis should remain rule-based or later become a classical ML model.
6. Decide how strict the `dangerous` label should be, because stego images are not always malicious by themselves.

## 15. Recommended Next Step

Start by implementing the installable CLI and metadata/file validation baseline. This gives the project an immediate working security component, creates useful features before the CNN is trained, and establishes the same analysis pipeline that the future API can reuse.

Recommended first code task:

```text
Create an installable Python CLI command that accepts an image path and outputs:
- file hash
- detected file type
- image dimensions
- extracted metadata
- suspicious indicators
- metadata risk score
- preliminary label: safe, suspicious, or dangerous
```

After that, the CNN training pipeline can be added around a structured dataset.
