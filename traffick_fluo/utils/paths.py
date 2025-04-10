# traffick_fluo/utils/paths.py

import os

def get_stage_dir(stage: str) -> str:
    """
    Get versioned subdirectory name for a pipeline stage.

    Parameters
    ----------
    stage : str
        One of: "segmentation", "membrane", "transfection".

    Returns
    -------
    str
        Directory name in format "01_segmentation", "02_membrane", etc.

    Raises
    ------
    ValueError
        If stage is not recognized.
    """
    order = {
        "segmentation": 1,
        "membrane": 2,
        "transfection": 3,
    }.get(stage)

    if order is None:
        raise ValueError(f"Unknown stage: {stage}")
    return f"{order:02d}_{stage}"


def get_model_root(run_id: str, stage: str) -> str:
    """
    Construct model root directory path for a given run and stage.

    Parameters
    ----------
    run_id : str
        ID of the run (e.g. "001").
    stage : str
        Stage name.

    Returns
    -------
    str
        Full path to model root directory.
    """
    return os.path.join("outputs", run_id, get_stage_dir(stage), "models")


def get_next_model_version_path(stage_dir: str) -> str:
    """
    Determine the next available model version directory.

    Parameters
    ----------
    stage_dir : str
        Path to the model root directory.

    Returns
    -------
    str
        Version string like "001", "002", etc.
    """
    if not os.path.exists(stage_dir):
        return "001"

    existing = [
        int(name)
        for name in os.listdir(stage_dir)
        if os.path.isdir(os.path.join(stage_dir, name)) and name.isdigit()
    ]
    next_version = max(existing, default=0) + 1
    return f"{next_version:03d}"


def find_latest_config(run_id: str, stage: str) -> str:
    """
    Locate the latest config.yaml for a given run and stage.

    Parameters
    ----------
    run_id : str
        Pipeline run identifier.
    stage : str
        Stage name.

    Returns
    -------
    str
        Path to the most recent config.yaml file.

    Raises
    ------
    FileNotFoundError
        If no model versions exist for the given stage.
    """
    model_root = get_model_root(run_id, stage)
    versions = sorted(
        [v for v in os.listdir(model_root) if v.isdigit()],
        key=lambda x: int(x)
    )
    if not versions:
        raise FileNotFoundError(f"No model versions found in {model_root}")
    latest_version = versions[-1]
    return os.path.join(model_root, latest_version, "config.yaml")
