# traffick_fluo/utils/metadata.py

import platform
from datetime import datetime
from typing import Dict, Any, List, Optional


def generate_metadata(
    config: Dict[str, Any],
    version: str,
    input_files: List[str],
    model_info: Dict[str, Any],
    score_summary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a structured metadata dictionary to accompany a trained model.

    Parameters
    ----------
    config : dict
        Parsed YAML config used for training.
    version : str
        Version string for the current model (e.g. "001").
    input_files : list of str
        List of input feature files used for training/scoring.
    model_info : dict
        Dictionary with model type and parameters. Expected keys:
            - 'type': model name
            - 'params': dict of hyperparameters
    score_summary : dict, optional
        Dictionary with evaluation metrics (e.g., AUC, F1). Default is empty.

    Returns
    -------
    dict
        Metadata dictionary ready for JSON serialization.
    """
    return {
        "stage": config["stage"],
        "run_id": config["run_id"],
        "version": version,
        "input_files": input_files,
        "features": list(model_info.get("params", {}).keys()),
        "model": {
            "type": model_info["type"],
            "params": model_info.get("params", {}),
        },
        "score_summary": score_summary or {},
        "input_hashes": {},  # Placeholder for future reproducibility
        "env": {
            "python": platform.python_version(),
            "sklearn": _get_version("sklearn"),
            "cellpose": _get_version("cellpose"),
        },
        "source": {
            "parent_stage": config.get("provenance", {}).get("parent_stage"),
            "parent_run_id": config.get("provenance", {}).get("parent_run_id"),
            "parent_version": config.get("provenance", {}).get("parent_version"),
        },
        "timestamp": datetime.now().isoformat(),
    }


def _get_version(pkg_name: str) -> str:
    """
    Safely fetch the __version__ string of an installed package.

    Parameters
    ----------
    pkg_name : str
        Importable name of the package (e.g., "sklearn", "cellpose").

    Returns
    -------
    str
        Version string or "unknown" if not available.
    """
    try:
        mod = __import__(pkg_name)
        return mod.__version__
    except Exception:
        return "unknown"

