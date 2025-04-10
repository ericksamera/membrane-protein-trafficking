# traffick_fluo/core/sample.py

import json
import numpy as np
import os
from typing import Optional, Dict, List, Any

from traffick_fluo.utils.logging import logger


class SampleCell:
    def __init__(
        self,
        cell_id: str,
        image_id: str,
        features: Optional[Dict[str, Any]] = None,
        scores: Optional[Dict[str, float]] = None,
        label: Optional[int] = None,
    ):
        """
        Initialize a SampleCell object that holds data about a single cell.

        Parameters
        ----------
        cell_id : str
            Unique identifier for the cell.
        image_id : str
            The image from which the cell was extracted.
        features : dict, optional
            Dictionary of extracted features (e.g., area, intensity).
        scores : dict, optional
            Dictionary of computed scores (e.g., segmentation_score).
        label : int or None, optional
            Binary label for classification (1 for good, 0 for bad).
        """
        self.cell_id = cell_id
        self.image_id = image_id
        self.features = features or {}
        self.scores = scores or {}
        self.label = label

    def build_feature_record(self, extra: Dict[str, Any] = None, score_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Construct a dictionary suitable for a row in the features.csv file.

        Parameters
        ----------
        extra : dict, optional
            Additional features to include in the record.
        score_key : str, optional
            If provided, uses self.scores[score_key] if available.

        Returns
        -------
        dict
            Row with metadata and features.
        """
        record = {
            "cell_id": self.cell_id,
            "image": self.image_id,
            **self.features,
        }

        if score_key and score_key in self.scores:
            record[score_key] = self.scores[score_key]

        if self.label is not None:
            record["label"] = self.label
        else:
            record["label"] = None

        if extra:
            record.update(extra)

        return record


    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the SampleCell to a dictionary suitable for JSON serialization.

        Returns
        -------
        dict
            Dictionary containing cell data.
        """
        return {
            "label": self.label,
            "cell_id": self.cell_id,
            "image_id": self.image_id,
            "features": self.features,
            "scores": self.scores,
        }

    def set_crop_box(self, x0: int, x1: int, y0: int, y1: int) -> None:
        """
        Set the crop bounding box coordinates in the feature dictionary.

        Parameters
        ----------
        x0, x1, y0, y1 : int
            Pixel coordinates of the crop bounds.
        """
        self.features.update({"x0": x0, "x1": x1, "y0": y0, "y1": y1})


    def set_scores(self, **kwargs: float) -> None:
        """
        Set classification scores for the cell.

        Parameters
        ----------
        **kwargs : float
            Keyword arguments mapping score names to float values (e.g., transfection_score=0.95).
        """
        self.scores = kwargs


    def set_features(self, base: Dict[str, Any], **kwargs) -> None:
        """
        Set features dictionary by combining a base dict with additional key-values.

        Parameters
        ----------
        base : dict
            Base feature dictionary (e.g., from previous stage).
        **kwargs : any
            Additional per-stage features to add or overwrite.
        """
        self.features = {**base, **kwargs}



    @staticmethod
    def save(cells: List["SampleCell"], output_dir: str) -> None:
        """
        Save a list of SampleCell objects to a JSON file.

        Parameters
        ----------
        cells : list of SampleCell
            The cell instances to serialize.
        output_dir : str
            Path to directory where JSON file will be written.
        """
        cell_dicts = [cell.to_dict() for cell in cells]
        output_file = os.path.join(output_dir, "per_cell_data.json")
        with open(output_file, "w") as f:
            json.dump(cell_dicts, f, indent=4)
        logger.info(f"Saved {len(cells)} cells' data to {output_file}")

    @staticmethod
    def load(input_path: str) -> List["SampleCell"]:
        """
        Load SampleCell objects from a saved JSON file.

        Parameters
        ----------
        input_path : str
            Path to the input JSON file.

        Returns
        -------
        list of SampleCell
            Deserialized SampleCell instances.
        """
        with open(input_path, "r") as f:
            data = json.load(f)
        return [SampleCell(**cell_data) for cell_data in data]

    @staticmethod
    def from_dataframe_row(row: Any, score_key: Optional[str] = None) -> "SampleCell":
        """
        Reconstruct a SampleCell from a row in features.csv.

        Parameters
        ----------
        row : pd.Series
            A single row from the features DataFrame.
        score_key : str, optional
            Name of the score column (e.g., "membrane_score") to extract.

        Returns
        -------
        SampleCell
            Reconstructed cell object.
        """
        drop_keys = {"cell_id", "image", "label", "crop_image"}
        if score_key:
            drop_keys.add(score_key)

        features = {k: row[k] for k in row.index if k not in drop_keys}
        cell = SampleCell(
            cell_id=row["cell_id"],
            image_id=row["image"],
            features=features,
            label=row.get("label")
        )
        if score_key and score_key in row:
            cell.set_scores(**{score_key: row[score_key]})
        return cell


    def save_masks(self, masks: Dict[str, np.ndarray], output_dir: str) -> None:
        """
        Save one or more binary masks for the cell (e.g., 'cytosol', 'membrane').

        Parameters
        ----------
        masks : dict[str, np.ndarray]
            Mapping of mask name to binary mask array.
        output_dir : str
            Directory where the masks should be saved.
        """
        for name, mask in masks.items():
            path = os.path.join(output_dir, f"{self.cell_id}_{name}_mask.npy")
            np.save(path, mask)
            self.features[f"{name}_mask_path"] = os.path.basename(path)

    def load_masks(self, crops_dir: str) -> Dict[str, np.ndarray]:
        """
        Load all masks associated with this cell using paths stored in `features`.

        Parameters
        ----------
        crops_dir : str
            Path to the crops directory containing mask files.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of loaded mask arrays.
        """
        loaded = {}
        for key in self.features:
            if key.endswith("_mask_path"):
                name = key.replace("_mask_path", "")
                path = os.path.join(crops_dir, self.features[key])
                if os.path.exists(path):
                    loaded[name] = np.load(path)
        return loaded
