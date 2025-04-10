# Cell Image Classification Pipeline

**Quantifying membrane protein trafficking phenotypes from microscopy images using a modular, stage-based bioimage analysis pipeline.**

---

## Overview

This repository provides a reproducible pipeline for analyzing dual-channel fluorescent microscopy images of single cells undergoing membrane protein trafficking. It supports high-throughput image segmentation, per-cell feature extraction, and classification via machine learning. The pipeline is tailored to membrane trafficking studies and supports rigorous versioning, threshold-based filtering, and metadata tracking.

Biological context:
Membrane protein trafficking is a vital process in cell signaling, development, and disease. This pipeline enables automated classification of trafficking states across thousands of individual cells, accelerating biological discovery.

---

## Pipeline Architecture

The pipeline is strictly ordered into three sequential stages:

1. **Segmentation**
   - Detects and isolates individual cells using Cellpose.
   - Extracts morphological and intensity features.
   - Saves binary masks, image crops, and metadata.

2. **Membrane Assessment**
   - Derives ring and inner masks from segmentation outputs.
   - Computes geometric, intensity, and texture-based membrane features.
   - Classifies and filters cells based on membrane quality.

3. **Transfection Scoring**
   - Quantifies protein signal in cytosolic and membrane compartments.
   - Produces binary and continuous transfection scores.
   - Exports annotated overlays and final features for high-confidence cells.

Each stage filters for high-confidence cells, passing only those to the next stage.

---

## Installation

Clone the repository and set up a Python 3.8+ environment:

```bash
git clone https://github.com/ericksamera/membrane-protein-trafficking.git
cd membrane-protein-trafficking
pip install -r requirements.txt
```

Recommended: Use `conda` or `virtualenv` for environment isolation.

---

## Input Format

Input images must be dual-channel with the following layout:

- **Channel 0**: Protein of interest (e.g., transfection marker)
- **Channel 1**: Membrane marker

Supported formats:
- `.tif`
- `.npy` with shape `[2, height, width]`

Images are specified via YAML config files.

---

## Usage

All pipeline stages are executed through the centralized CLI:

```bash
python traffick_fluo.py <command> --config <path/to/config.yaml>
```

### Available Commands

- `segment`: Run segmentation and extract features
- `prepare`: Prepare inputs for membrane or transfection stages
- `train`: Train a model for a stage using labeled features
- `score`: Apply a trained model to classify cells
- `rescore`: Retrain and score using previously extracted features

### Example Commands

```bash
python traffick_fluo.py segment --config configs/segment.yaml
python traffick_fluo.py prepare --config configs/membrane.yaml
python traffick_fluo.py train --config configs/membrane.yaml
python traffick_fluo.py score --config configs/transfection.yaml
```

Each command reads a structured YAML config and saves outputs under a versioned hierarchy.

---

## Output Structure

All outputs are grouped by `run_id`, with per-stage folders:

```
outputs/
└── 001/
    ├── 01_segmentation/
    │   ├── crops/
    │   └── models/
    ├── 02_membrane/
    │   └── crops/
    ├── 03_transfection/
    │   └── crops/
    └── images/
```

Each `crops/` folder contains:
- Per-cell overlay images
- Binary masks
- `features.csv`: extracted feature matrix
- `per_cell_data.json`: metadata for each cell

Each `models/` subdirectory is versioned (`001/`, `002/`, etc.) and contains:
- `model.pkl`: serialized trained model
- `scored.csv`: classification results
- `config.yaml`: full configuration for the run
- `metadata.json`: environment, input hashes, threshold, and provenance
- `log.txt`: execution log

---

## Reproducibility

This pipeline ensures full reproducibility through:
- YAML-driven configuration for all processing
- Auto-incremented model versioning
- Preservation of all model outputs and thresholds
- Metadata tracking of environment, parameters, and inputs

No files are overwritten between runs.

---

## License

This project is licensed under the MIT License. Academic and commercial use is permitted with proper attribution.
