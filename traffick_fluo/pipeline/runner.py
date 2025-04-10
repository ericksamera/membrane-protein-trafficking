# traffick_fluo/pipeline/runner.py

import os
import yaml
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from skimage.io import imsave, imread
from skimage.measure import regionprops

from traffick_fluo.core.sample import SampleCell
from stages import STAGE_REGISTRY
from traffick_fluo.utils.paths import get_stage_dir
from traffick_fluo.utils.prep import prepare_stage_with_filtering
from traffick_fluo.model.train import train_model

from traffick_fluo.utils.logging import logger


# === Utility ===
def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize an image to uint8 format (0–255)."""
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max() if img.max() != 0 else 1
    return (img * 255).astype(np.uint8)


def save_image_channels(img: np.ndarray, img_index: str, orig_filename: str, images_dir: str) -> None:
    """Save RGB channel visualizations for protein, membrane, and overlay."""
    imsave(os.path.join(images_dir, f"{img_index}__{orig_filename}"), img)

    protein_rgb = np.zeros((*img[0].shape, 3), dtype=np.uint8)
    protein_rgb[:, :, 1] = to_uint8(img[0])
    imsave(os.path.join(images_dir, f"{img_index}_0_protein.png"), protein_rgb)

    membrane_scaled = to_uint8(img[1])
    membrane_rgb = np.zeros((*img[1].shape, 3), dtype=np.uint8)
    membrane_rgb[:, :, 0] = membrane_rgb[:, :, 2] = membrane_scaled
    imsave(os.path.join(images_dir, f"{img_index}_1_membrane.png"), membrane_rgb)

    overlay = np.zeros((*img[0].shape, 3), dtype=np.uint8)
    overlay[:, :, 1] = to_uint8(img[0])
    overlay[:, :, 0] = overlay[:, :, 2] = membrane_scaled
    imsave(os.path.join(images_dir, f"{img_index}_9_overlay.png"), overlay)


# === Segment ===
def run_segment(config_path: str) -> None:
    """
    Run segmentation stage:
        - Applies Cellpose
        - Extracts regions and crops
        - Computes features
        - Saves images, masks, and metadata
    """
    config = load_config(config_path)
    run_id = config["run_id"]
    output_root = config["scoring"]["output_dir"]
    image_paths = config["input"]["images"]

    stage = STAGE_REGISTRY["segmentation"]()
    stage_dir = get_stage_dir("segmentation")
    stage_output_dir = os.path.join(output_root, run_id, stage_dir)
    images_dir = os.path.join(output_root, run_id, "images")
    crops_dir = os.path.join(stage_output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    all_cells, all_regions, all_context = [], [], []

    for i, img_path in enumerate(image_paths):
        img = imread(img_path)
        img_index = f"{i:0{len(str(len(image_paths)))}d}"
        save_image_channels(img, img_index, os.path.basename(img_path), images_dir)

        masks, *_ = stage.model.eval(img, channels=[0, 0], diameter=None)
        props = regionprops(masks, intensity_image=img[1])
        image_id = os.path.basename(img_path)

        for region in props:
            all_cells.append(SampleCell(cell_id=None, image_id=image_id))
            all_regions.append(region)
            all_context.append((masks, img))

    for i, cell in enumerate(all_cells):
        cell.cell_id = str(i)

    feature_rows = []
    for cell, region, (masks, img) in zip(all_cells, all_regions, all_context):
        try:
            crop_img, crop_overlay, (y0, x0, y1, x1), crop_mask = stage._render_crop(img, masks, region)
        except Exception as e:
            logger.warning(f"Skipping cell {cell.cell_id}: {e}")
            continue

        crop_fn = f"{cell.cell_id}.png"
        mask_fn = f"{cell.cell_id}_mask.npy"

        imsave(os.path.join(crops_dir, crop_fn), crop_overlay)
        np.save(os.path.join(crops_dir, mask_fn), crop_mask)

        features = {f.__name__: f(region) for f in stage.FEATURE_FUNCS}
        features.update({
            "crop_image": crop_fn,
            "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "mask_path": mask_fn
        })

        cell.features = features
        feature_rows.append({
            "label": None,
            "cell_id": cell.cell_id,
            "image": cell.image_id,
            **features,
            "segmentation_score": 1.0
        })

    pd.DataFrame(feature_rows).to_csv(os.path.join(stage_output_dir, "features.csv"), index=False)
    SampleCell.save(all_cells, crops_dir)
    logger.info(f"✅ Segment: Saved {len(all_cells)} cells with overlays, masks, and features.")


# === Train ===
def run_train(
    config_path: Optional[str] = None,
    rescore: bool = False,
    run_id: Optional[str] = None,
    stage: Optional[str] = None
) -> None:
    """
    Train or retrain a model on features from a given stage.

    Parameters
    ----------
    config_path : str
        Path to training YAML config.
    rescore : bool
        Whether to retrain on existing scored features.
    run_id : str, optional
        Optional override for run ID (not used here).
    stage : str, optional
        Optional override for stage (not used here).
    """
    if not config_path:
        raise ValueError("Must provide --config")

    if rescore:
        config = load_config(config_path)
        config["input"]["from_scored"] = True
    else:
        config = _load_config_for_training(config_path)

    train_model(config)


def _load_config_for_training(config_path: str) -> Dict[str, Any]:
    """
    Load config and run `prepare_stage_with_filtering` if needed.

    Auto-prepares membrane/transfection input if not explicitly passed.
    """
    config = load_config(config_path)
    stage = config["stage"]
    if stage in ["membrane", "transfection"] and "features_csv" not in config.get("input", {}):
        parent = "segmentation" if stage == "membrane" else "membrane"
        prepare_stage_with_filtering(config, parent_stage_name=parent, current_stage_name=stage)
    return config


# === Prepare ===
def run_prepare(config_path: str) -> None:
    """
    Prepare input features and masks for membrane or transfection stage.
    """
    config = load_config(config_path)
    stage = config["stage"]

    if stage not in ["membrane", "transfection"]:
        raise ValueError(f"Stage '{stage}' does not support preparation — use `segment` instead.")

    parent = "segmentation" if stage == "membrane" else "membrane"
    prepare_stage_with_filtering(config, parent_stage_name=parent, current_stage_name=stage)


# === Score ===
def run_score(config_path: str) -> None:
    """
    Score features using the latest trained model for a given stage.
    """
    config = load_config(config_path)
    stage_name = config["stage"]

    if stage_name not in STAGE_REGISTRY:
        raise ValueError(f"Unknown stage: {stage_name}")

    stage = STAGE_REGISTRY[stage_name]()
    stage.score(config)
    logger.info(f"✅ Score: Finished scoring for stage '{stage_name}' using config '{config_path}'")
