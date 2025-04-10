# traffick_fluo/model/train.py

import os
import joblib
import yaml
import json
from datetime import datetime
from typing import Any, Dict

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score

from traffick_fluo.utils.io import load_features, reorder_feature_columns
from traffick_fluo.utils.paths import get_next_model_version_path, get_stage_dir
from traffick_fluo.utils.metadata import generate_metadata
from traffick_fluo.utils.logging import logger
from traffick_fluo.model.registry import MODEL_REGISTRY


def train_model(config: Dict[str, Any]) -> None:
    """
    Train a model on labeled features with optional cross-validation.
    Applies scoring, selects best threshold, saves model and outputs.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration with keys:
            - run_id
            - stage
            - input.features_csv
            - input.labels_column
            - model.type
            - model.params
            - model.cross_validation
            - scoring.output_dir

    Outputs
    -------
    Saves:
        - model.pkl
        - features.csv (with scores and passed_filter)
        - scored.csv
        - config.yaml
        - metadata.json
        - log.txt
    """
    full_df = load_features(config["input"]["features_csv"])
    full_df["cell_id"] = full_df["cell_id"].astype(str)
    from_scored = config["input"].get("from_scored", False)

    if from_scored:
        full_df.drop(columns=["passed_filter"], errors="ignore", inplace=True)

    label_col = config["input"].get("labels_column", "label")
    labeled_df = full_df.dropna(subset=[label_col])

    drop_cols = ["cell_id", "image", "crop_image", label_col, "mask_path", "x0", "x1", "y0", "y1"]
    X_train = labeled_df.drop(columns=[c for c in drop_cols if c in labeled_df.columns])
    X_train = X_train.select_dtypes(include=["number"])
    feature_cols = X_train.columns.tolist()

    y_train = labeled_df[label_col]

    # Instantiate model
    model_cfg = config["model"]
    model_type = model_cfg["type"]
    model_params = model_cfg.get("params", {})
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    ModelClass = MODEL_REGISTRY[model_type]
    model = ModelClass(**model_params)

    # Optional cross-validation
    cv_cfg = model_cfg.get("cross_validation", {})
    do_cv = cv_cfg.get("enabled", False)
    best_model = None
    best_score = -1.0
    scores = []

    if do_cv:
        folds = cv_cfg.get("folds", 5)
        stratify = cv_cfg.get("stratify", True)
        cv = StratifiedKFold(n_splits=folds) if stratify else KFold(n_splits=folds)
        for train_idx, val_idx in cv.split(X_train, y_train):
            clf = ModelClass(**model_params)
            clf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            y_prob = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
            score = roc_auc_score(y_train.iloc[val_idx], y_prob)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_model = clf
    else:
        model.fit(X_train, y_train)
        best_model = model

    # Prepare paths
    run_id = str(config["run_id"]).zfill(3)
    stage = config["stage"]
    output_root = config["scoring"]["output_dir"]
    stage_dir = get_stage_dir(stage)
    model_version = get_next_model_version_path(os.path.join(output_root, run_id, stage_dir, "models"))
    model_dir = os.path.join(output_root, run_id, stage_dir, "models", model_version)
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    joblib.dump(best_model, os.path.join(model_dir, "model.pkl"))

    # Score all cells (labeled + unlabeled)
    X_all = full_df[feature_cols]
    score_col = f"{stage}_score"
    full_df[score_col] = best_model.predict_proba(X_all)[:, 1]

    # Optimize threshold using labeled set
    y_prob = full_df.loc[full_df[label_col].notna(), score_col]
    y_true = full_df.loc[full_df[label_col].notna(), label_col]
    thresholds = sorted(set(y_prob))
    best_thresh = 0.5
    best_f1 = 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Apply filtering
    full_df["passed_filter"] = (full_df[score_col] >= best_thresh).astype(int)
    config["scoring"]["threshold"] = best_thresh

    full_df = reorder_feature_columns(full_df, score_col=score_col)

    # Save outputs
    scored_csv_path = os.path.join(model_dir, "scored.csv")
    full_df.to_csv(scored_csv_path, index=False)
    config["scoring"]["scored_csv"] = scored_csv_path

    with open(os.path.join(model_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    metadata = generate_metadata(
        config=config,
        version=model_version,
        input_files=[config["input"]["features_csv"]],
        model_info={"type": model_type, "params": model_params},
        score_summary={"auc": best_score if scores else None},
    )
    metadata["threshold"] = best_thresh
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    with open(os.path.join(model_dir, "log.txt"), "w") as f:
        f.write(f"Training completed at {datetime.now().isoformat()}\n")
        if scores:
            f.write(f"Cross-validation AUCs: {scores}\n")
            f.write(f"Best AUC: {best_score:.4f}\n")
        f.write(f"Best threshold (F1 optimized): {best_thresh:.3f} with F1={best_f1:.4f}\n")

    logger.info(f"✅ Training complete. Model saved to: {model_dir}")
