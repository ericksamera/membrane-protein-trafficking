# traffick_fluo/utils/prep.py

import os
import json
import yaml
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any

from traffick_fluo.core.sample import SampleCell
from traffick_fluo.utils.paths import find_latest_config
from traffick_fluo.pipeline.registry import STAGE_REGISTRY
from traffick_fluo.utils.logging import logger


def get_image_path(config: Dict[str, Any]) -> str:
    """
    Extract the primary image path from the config.

    Parameters
    ----------
    config : dict
        The parsed YAML configuration.

    Returns
    -------
    str
        Path to the first image.

    Raises
    ------
    ValueError
        If no valid image path is provided.
    """
    image_path = config.get("input", {}).get("image_path")
    if not image_path:
        images = config.get("input", {}).get("images", [])
        if images:
            image_path = images[0]
        else:
            raise ValueError("Missing image path in config['input']")
    return image_path


def load_parent_metadata(output_root: str, run_id: str, parent_stage: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Load per-cell metadata JSON from the parent stage's output.

    Parameters
    ----------
    output_root : str
        Root output directory (typically 'outputs').
    run_id : str
        Pipeline run ID.
    parent_stage : str
        Name of the parent stage ("segmentation" or "membrane").

    Returns
    -------
    tuple
        - parent_dir : str — directory name of the parent stage
        - metadata : dict[str, dict] — cell_id → cell metadata

    Raises
    ------
    FileNotFoundError
        If the per-cell JSON does not exist.
    """
    parent_dir = f"{int(STAGE_REGISTRY[parent_stage].ORDER):02d}_{parent_stage}"
    path = os.path.join(output_root, run_id, parent_dir, "crops", "per_cell_data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing per-cell data at: {path}")
    with open(path) as f:
        return parent_dir, {cell["cell_id"]: cell for cell in json.load(f)}


def resolve_passed_cell_ids(
    config: Dict[str, Any],
    output_root: str,
    parent_stage: str,
    run_id: str,
    parent_version: Optional[str] = None
) -> Optional[List[str]]:
    """
    Load scored.csv and determine which cell_ids passed the filter.

    Parameters
    ----------
    config : dict
        Configuration used to extract run metadata.
    output_root : str
        Base output directory path.
    parent_stage : str
        Name of the upstream stage.
    run_id : str
        Pipeline run ID (typically same as current).
    parent_version : str, optional
        Optional override to use a specific model version.

    Returns
    -------
    list[str] or None
        List of passed cell_ids, or None if fallback is required.
    """
    try:
        if parent_version:
            cfg_path = os.path.join(
                output_root, run_id,
                f"{int(STAGE_REGISTRY[parent_stage].ORDER):02d}_{parent_stage}",
                "models", parent_version, "config.yaml"
            )
        else:
            cfg_path = find_latest_config(run_id, parent_stage)

        with open(cfg_path) as f:
            parent_cfg = yaml.safe_load(f)

        scored_csv = parent_cfg.get("scoring", {}).get("scored_csv")
        if not scored_csv or not os.path.exists(scored_csv):
            raise FileNotFoundError(f"Missing scored.csv at: {scored_csv}")

        df = pd.read_csv(scored_csv)
        if "passed_filter" not in df.columns:
            raise ValueError("Missing 'passed_filter' column in scored.csv")

        passed = df[df["passed_filter"] == 1]["cell_id"].astype(str).tolist()
        logger.info(f"{len(passed)} cells passed filtering from {parent_stage}")
        return passed

    except Exception as e:
        logger.warning(f"[Fallback] Using all cells from parent stage due to error: {e}")
        return None


def prepare_stage_with_filtering(
    config: Dict[str, Any],
    parent_stage_name: str,
    current_stage_name: str
) -> None:
    """
    Prepare feature inputs for a downstream stage using high-confidence cells from the parent stage.

    Parameters
    ----------
    config : dict
        Parsed YAML config with stage info and paths.
    parent_stage_name : str
        Name of the upstream stage (e.g., "segmentation").
    current_stage_name : str
        Name of the current stage to prepare (e.g., "membrane", "transfection").

    Side Effects
    ------------
    - Saves filtered features to features.csv in the stage directory.
    - Saves per-cell metadata to per_cell_data.json.
    """
    run_id = config["run_id"]
    output_root = config.get("output", {}).get("root") \
        or config.get("scoring", {}).get("output_dir") \
        or "outputs"
    image_path = get_image_path(config)

    provenance = config.get("provenance", {})
    parent_stage = provenance.get("parent_stage", parent_stage_name)
    parent_run_id = provenance.get("parent_run_id", run_id)
    parent_version = provenance.get("parent_version")

    # Load metadata from parent stage
    parent_dir, per_cell = load_parent_metadata(output_root, parent_run_id, parent_stage)
    passed_ids = list(per_cell.keys())  # fallback to all

    # Try using passed_filter from scored.csv
    filtered_ids = resolve_passed_cell_ids(config, output_root, parent_stage, parent_run_id, parent_version)
    if filtered_ids:
        passed_ids = filtered_ids

    # Extract features for passed cells only
    stage = STAGE_REGISTRY[current_stage_name]()
    stage_dir = f"{int(stage.ORDER):02d}_{current_stage_name}"
    stage_output_dir = os.path.join(output_root, run_id, stage_dir)
    crops_dir = os.path.join(stage_output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    df = stage.extract_features(
        image_path=image_path,
        filter_cell_ids=passed_ids,
        output_dir=stage_output_dir
    )

    score_key = "membrane_score" if current_stage_name == "transfection" else "membrane_score"
    cells = [SampleCell.from_dataframe_row(row, score_key=score_key) for _, row in df.iterrows()]
    return None