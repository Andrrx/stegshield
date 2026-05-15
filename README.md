# StegShield

StegShield is a bachelor thesis project for detecting suspicious and dangerous image files using a hybrid approach:

- CNN-based image analysis.
- Metadata and file-structure inspection.
- A fusion layer that labels files as `safe`, `suspicious`, or `dangerous`.

The first version will be an installable command-line tool:

```bash
stegshield analyze ./image.png
```

Later versions will expose the same analysis pipeline through an API so applications can scan uploaded images, such as profile pictures or user-generated content.

## Current Status

Planning and repository setup.

## Planned Features

- File type validation.
- Hashing and file metadata extraction.
- EXIF and structural risk indicators.
- Custom CNN trained from scratch.
- Optional comparison against pretrained CNN models.
- Combined CNN + metadata decision layer.
- CLI output in human-readable and JSON formats.
- Future FastAPI integration.

## Labels

- `safe`: no suspicious indicators.
- `suspicious`: weak or moderate risk indicators.
- `dangerous`: strong evidence of payload-like behavior, steganographic embedding, malformed structure, or multiple high-risk indicators.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the placeholder CLI:

```bash
stegshield --help
```

Run tests:

```bash
pytest
```

## Dataset Direction

Candidate datasets include:

- Kaggle Stego Images Dataset: https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset
- StegoAppDB: https://forensicstats.org/stegoappdb/
- BOSSBase / BOWS2-based steganalysis datasets.
- ALASKA steganalysis datasets.

Real malicious files should not be executed. Suspicious samples must be handled defensively and, if necessary, in an isolated environment.
