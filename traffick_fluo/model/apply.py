# traffick_fluo/model/apply.py

import os
import joblib
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional

from traffick_fluo.utils.io import load_features, reorder_feature_columns
from traffick_fluo.utils.logging import logger
from traffick_fluo.utils.paths import get_stage_dir


def score_features(stage: Any, config: Dict[str, Any]) -> None:
    """
    Apply a trained model to score features and filter high-confidence cells.

    Parameters
    ----------
    stage : object
        Stage object (e.g., SegmentationStage, MembraneStage) that implements scoring.
    config : dict
        YAML-parsed configuration dictionary with keys:
            - run_id : str
            - stage : str
            - scoring.output_dir : str
            - model_version : Optional[str]
            - scoring.threshold : Optional[float]

    Outputs
    -------
    - Saves scored CSV to: outputs/<run_id>/<stage_dir>/<version>/scored.csv
    - Appends log to: outputs/<run_id>/<stage_dir>/<version>/log.txt
    - Logs cell count summary via pipeline logger
    """
    run_id: str = config["run_id"]
    stage_name: str = config["stage"]
    output_root: str = config["scoring"]["output_dir"]
    stage_dir: str = get_stage_dir(stage_name)
    model_version: Optional[str] = config.get("model_version")

    # Resolve model path
    stage_path = os.path.join(output_root, run_id, stage_dir)
    if model_version is None:
        versions = sorted([d for d in os.listdir(stage_path) if d.isdigit()])
        if not versions:
            raise FileNotFoundError(f"No model version found in {stage_path}")
        model_version = versions[-1]

    model_path = os.path.join(stage_path, model_version, "model.pkl")
    features_path = os.path.join(stage_path, "features.csv")
    out_path = os.path.join(stage_path, model_version, "scored.csv")

    # Load model and features
    model = joblib.load(model_path)
    features_df = load_features(features_path)

    # Score: drop non-numeric/non-feature columns
    X = features_df.drop(columns=["cell_id", "image", "crop_image", "label", "passed_filter"], errors="ignore")
    score_col = f"{stage_name}_score"
    features_df[score_col] = model.predict_proba(X)[:, 1]

    # Apply classification threshold
    threshold: float = config["scoring"].get("threshold", 0.5)
    features_df["passed_filter"] = (features_df[score_col] >= threshold).astype(int)

    # Save scored output
    features_df = reorder_feature_columns(features_df, score_col=score_col)
    features_df.to_csv(out_path, index=False)

    # Log summary
    passed_count = int(features_df["passed_filter"].sum())
    total = len(features_df)
    logger.info(f"Scoring complete: {passed_count}/{total} cells passed ({passed_count / total:.1%})")

    log_path = os.path.join(stage_path, model_version, "log.txt")
    with open(log_path, "a") as f:
        f.write(f"Scored {total} cells at {datetime.now().isoformat()}\n")
        f.write(f"Threshold = {threshold:.3f} | Passed = {passed_count}/{total}\n")
