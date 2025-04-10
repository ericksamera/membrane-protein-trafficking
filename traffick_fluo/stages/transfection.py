# traffick_fluo/stages/transfection.py

from .base import BaseStage

import os
import json
import pandas as pd
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu

from typing import Optional, List

# Project modules
from traffick_fluo.core.sample import SampleCell
from traffick_fluo.core.overlay import green_overlay_with_contours, magenta_overlay
from traffick_fluo.core.crop_utils import extract_crop_bounds, crop_image
from traffick_fluo.utils.logging import logger
from traffick_fluo.utils.feature_registry import create_feature_registry

# Feature registry
feature, FEATURE_FUNCS = create_feature_registry()

# === FEATURE FUNCTIONS ===

@feature
def signal_present(row): return row["signal_present"]

@feature
def signal_mean(row): return row["signal_mean"]

@feature
def signal_threshold(row): return row["signal_threshold"]

@feature
def transfection_score(row): return row["transfection_score"]

class TransfectionStage(BaseStage):
    ORDER = 3

    FEATURE_FUNCS = FEATURE_FUNCS

    def extract_features(
        self,
        image_path: str,
        *,
        filter_cell_ids: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:

        if not filter_cell_ids:
            raise RuntimeError("No cell IDs passed from membrane stage")

        run_root = os.path.dirname(output_dir)
        parent_dir = os.path.join(run_root, "02_membrane", "crops")
        with open(os.path.join(parent_dir, "per_cell_data.json")) as f:
            parent_cells = {c["cell_id"]: c for c in json.load(f)}

        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        records, sample_cells = [], []

        for cid in filter_cell_ids:
            meta = parent_cells.get(cid)
            if not meta:
                logger.warning(f"No metadata for cell {cid}")
                continue

            image_id = meta["image_id"]
            bounds = extract_crop_bounds(meta)
            if bounds is None:
                logger.warning(f"Missing crop bounds for cell {cid}")
                continue

            cell = SampleCell(cell_id=cid, image_id=image_id, features=meta["features"])
            masks = cell.load_masks(parent_dir)
            cytosol_mask = masks.get("cytosol")
            ring_mask = masks.get("membrane")
            if cytosol_mask is None or ring_mask is None:
                logger.warning(f"Missing masks for cell {cid}")
                continue

            try:
                full_img = imread(os.path.join("data", image_id))
                protein_img = full_img[0]
            except Exception as e:
                logger.warning(f"Error loading image for cell {cid}: {e}")
                continue

            crop = crop_image(protein_img, bounds)
            y0, y1, x0, x1 = bounds
            combined_mask = (cytosol_mask > 0) | (ring_mask > 0)
            signal_pixels = crop[combined_mask]

            if signal_pixels.size == 0:
                logger.warning(f"No signal pixels for cell {cid}")
                continue

            try:
                thresh = threshold_otsu(signal_pixels)
            except Exception:
                thresh = 0.0

            mean_val = signal_pixels.mean()
            present = int(mean_val > thresh)
            score_val = float(present)

            crop_image_str: str = f"{cid.lower()}.png"

            cell.set_crop_box(*bounds)
            cell.set_features(
                cell.features,
                signal_present=present,
                signal_mean=float(mean_val),
                signal_threshold=float(thresh),
                transfection_score=score_val,
                crop_image=crop_image_str
            )
            cell.set_scores(transfection_score=score_val)

            green = green_overlay_with_contours(crop, cytosol_mask, ring_mask)
            imsave(os.path.join(crops_dir, crop_image_str), green)

            membrane_crop = full_img[1][y0:y1, x0:x1]
            magenta = magenta_overlay(membrane_crop)
            imsave(os.path.join(crops_dir, f"{cid.lower()}_membrane.png"), magenta)

            records.append(cell.build_feature_record())
            sample_cells.append(cell)

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(output_dir, "features.csv"), index=False)
        SampleCell.save(sample_cells, crops_dir)
        logger.info(f"Transfection: Saved {len(df)} cells with overlays and heuristic scores.")
        return df


    def score(self, config):
        from traffick_fluo.model.apply import score_features
        score_features(self, config)
