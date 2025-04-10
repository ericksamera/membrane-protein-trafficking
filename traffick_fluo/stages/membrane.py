# traffick_fluo/stages/membrane.py

from .base import BaseStage

import os
import json
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import entropy as shannon_entropy
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops, label as sklabel
from skimage.morphology import (
    binary_opening,
    binary_closing,
    binary_erosion,
    remove_small_objects,
    disk,
)

from typing import Optional, List
import pandas as pd

# Project modules
from traffick_fluo.core.sample import SampleCell
from traffick_fluo.core.overlay import green_overlay_with_contours
from traffick_fluo.core.crop_utils import extract_crop_bounds, crop_image
from traffick_fluo.utils.logging import logger
from traffick_fluo.utils.feature_registry import create_feature_registry

# Feature registry
feature, FEATURE_FUNCS = create_feature_registry()

# ------------------------------------------------------------------------------
# Feature function registry
# ------------------------------------------------------------------------------

@feature
def membrane_score(row):
    return row["membrane_score"]

# --- scoring accessors: geometric ---

@feature
def outer_area(row): return row["outer_area"]

@feature
def inner_area(row): return row["inner_area"]

@feature
def ring_area(row): return row["ring_area"]

@feature
def area_ratio(row): return row["area_ratio"]

@feature
def ring_thickness_mean(row): return row["ring_thickness_mean"]

@feature
def ring_thickness_std(row): return row["ring_thickness_std"]

@feature
def solidity_outer(row): return row["solidity_outer"]

@feature
def solidity_inner(row): return row["solidity_inner"]

@feature
def eccentricity(row): return row["eccentricity"]

@feature
def compactness_outer(row): return row["compactness_outer"]

@feature
def compactness_inner(row): return row["compactness_inner"]

# --- scoring accessors: intensity ---

@feature
def mean_int_ring(row): return row["mean_int_ring"]

@feature
def std_int_ring(row): return row["std_int_ring"]

@feature
def mean_int_inner(row): return row["mean_int_inner"]

@feature
def mean_int_outer(row): return row["mean_int_outer"]

@feature
def contrast_ring_inner(row): return row["contrast_ring_inner"]

@feature
def contrast_ring_outer(row): return row["contrast_ring_outer"]

# --- scoring accessors: texture ---

@feature
def entropy_ring(row): return row["entropy_ring"]

@feature
def gradient_mean_ring(row): return row["gradient_mean_ring"]

@feature
def std_gradient_ring(row): return row["std_gradient_ring"]

@feature
def lbp_hist_mean(row): return row["lbp_hist_mean"]


# ------------------------------------------------------------------------------
# MembraneStage
# ------------------------------------------------------------------------------
class MembraneStage(BaseStage):
    """Stage‑2: membrane quality detection."""
    ORDER = 2

    FEATURE_FUNCS = FEATURE_FUNCS

    # ------------------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------------------

    @staticmethod
    def fallback_inner_by_erosion(outer_mask: np.ndarray, radius: int = 5) -> np.ndarray:
        """
        Return a shrunk version of the outer mask using morphological erosion.

        Parameters
        ----------
        outer_mask : binary np.ndarray
            Original mask from Cellpose.
        radius : int
            How many pixels to erode inward (controls ring thickness).

        Returns
        -------
        inner_mask : binary np.ndarray
            Eroded version of the mask (0/1).
        """
        inner = binary_erosion(outer_mask, disk(radius))
        return inner.astype(np.uint8)


    @staticmethod
    def segment_membrane(membrane_crop, *, sigma=1, min_size=30):
        img = gaussian(membrane_crop.astype(np.float32), sigma=sigma)
        thresh = threshold_otsu(img)
        binary = img > thresh
        clean = binary_opening(binary, disk(1))
        clean = remove_small_objects(clean, min_size=min_size)
        lbl = sklabel(clean)
        if lbl.max() == 0:
            return np.zeros_like(clean, dtype=np.uint8)
        largest = max(range(1, lbl.max() + 1), key=lambda x: (lbl == x).sum())
        return (lbl == largest).astype(np.uint8)

    @staticmethod
    def enforce_circular_inner(inner_mask: np.ndarray, outer_mask: np.ndarray, min_solidity: float = 0.8) -> np.ndarray:
        if inner_mask.sum() == 0:
            return inner_mask
        props = regionprops(inner_mask.astype(np.uint8))[0]
        solidity = props.solidity if props.solidity is not None else 1.0
        if solidity >= min_solidity:
            return inner_mask

        cy, cx = props.centroid
        dist = distance_transform_edt(outer_mask)
        radius = np.median(dist[inner_mask > 0])

        yy, xx = np.ogrid[:inner_mask.shape[0], :inner_mask.shape[1]]
        disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
        return disc.astype(np.uint8)

    def extract_membrane_ring(
        self,
        membrane_crop,
        cell_mask,
        *,
        inner_mode="percentile",
        percentile=40,
        shrink_fraction=0.5,
        refine="threshold",
        min_ring_area=25,
    ):
        dist = distance_transform_edt(cell_mask)
        if inner_mode == "percentile":
            target_r = np.percentile(dist[cell_mask > 0], percentile)
        else:
            target_r = dist.max() * shrink_fraction

        inner_mask = dist > target_r

        if refine == "threshold":
            inner_vals = membrane_crop[(cell_mask > 0) & inner_mask]
            if inner_vals.size:
                cyt_thresh = threshold_otsu(inner_vals)
                inner_mask &= membrane_crop < cyt_thresh
        elif refine == "gradient":
            grad = sobel(membrane_crop.astype(np.float32))
            med_grad = np.median(grad[cell_mask > 0])
            inner_mask &= grad < med_grad

        inner_mask = binary_closing(inner_mask, disk(2))
        inner_mask = binary_opening(inner_mask, disk(1))
        inner_mask = remove_small_objects(inner_mask, min_size=20)

        ring_mask = cell_mask & (~inner_mask)

        ring_area = ring_mask.sum()
        if ring_area < min_ring_area:
            inner_mask = self.fallback_inner_by_erosion(cell_mask, radius=5)
            ring_mask  = cell_mask & (~inner_mask)

        if ring_mask.sum() < min_ring_area:
            inner_mask[:] = 0
            ring_mask[:] = 0

        return ring_mask.astype(np.uint8), inner_mask.astype(np.uint8)

    # ------------------------------------------------------------------------------
    # Main pipeline API
    # ------------------------------------------------------------------------------

    def extract_features(
        self,
        image_path: str,
        *,
        filter_cell_ids: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        if not filter_cell_ids:
            raise RuntimeError("filter_cell_ids is empty — nothing to process.")

        run_root = os.path.dirname(output_dir)
        parent_dir = os.path.join(run_root, "01_segmentation", "crops")
        meta_path = os.path.join(parent_dir, "per_cell_data.json")
        with open(meta_path) as f:
            parent_cells = {c["cell_id"]: c for c in json.load(f)}

        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        records, sample_cells = [], []

        for cid in filter_cell_ids:
            meta = parent_cells.get(cid)
            if not meta:
                continue

            image_id = meta["image_id"]
            bounds = extract_crop_bounds(meta)
            if bounds is None:
                logger.warning(f"Missing crop bounds for cell {cid}")
                continue

            try:
                full_img = imread(os.path.join("data", image_id))
                membrane_img = full_img[1]
            except Exception as e:
                logger.info(f"Could not load image '{image_id}' for cell {cid}: {e}")
                continue

            crop = crop_image(membrane_img, bounds)
            y0, y1, x0, x1 = bounds

            mask_path = meta["features"].get("mask_path")
            if not mask_path:
                continue
            outer_mask = np.load(os.path.join(parent_dir, mask_path))

            ring_mask, inner_mask = self.extract_membrane_ring(crop, outer_mask)
            # ----------------------- feature computation -------------------
            # distance map for thickness + perimeter
            dist = distance_transform_edt(outer_mask)

            # --- areas ---
            outer_area = int(outer_mask.sum())
            inner_area = int(inner_mask.sum())
            ring_area  = int(ring_mask.sum())
            area_ratio = inner_area / outer_area if outer_area else 0.0

            # --- thickness statistics (pixels) ---
            ring_thickness_mean = dist[ring_mask].mean() if ring_mask.any() else 0.0
            ring_thickness_std  = dist[ring_mask].std()  if ring_mask.any() else 0.0

            # --- regionprops for shape ---
            props_outer = regionprops(outer_mask.astype(np.uint8))[0]
            solidity_outer    = props_outer.solidity
            eccentricity      = props_outer.eccentricity
            perimeter_outer   = props_outer.perimeter
            compactness_outer = (perimeter_outer**2) / (4 * np.pi * outer_area) if outer_area else 0.0

            # inner solidity (skip if empty)
            if inner_area:
                props_inner = regionprops(inner_mask.astype(np.uint8))[0]
                solidity_inner    = props_inner.solidity
                perimeter_inner   = props_inner.perimeter
                compactness_inner = (perimeter_inner**2) / (4 * np.pi * inner_area)
            else:
                solidity_inner = compactness_inner = 0.0

            # --- intensity statistics ---
            ring_vals  = crop[ring_mask]
            inner_vals = crop[inner_mask]
            outer_vals = crop[outer_mask]

            mean_ring  = ring_vals.mean()  if ring_vals.size else 0.0
            std_ring   = ring_vals.std()   if ring_vals.size else 0.0
            mean_inner = inner_vals.mean() if inner_vals.size else 0.0
            mean_outer = outer_vals.mean() if outer_vals.size else 0.0

            contrast_ring_inner = mean_ring - mean_inner
            contrast_ring_outer = mean_ring - mean_outer

            # --- texture features ---
            # Shannon entropy of ring pixels
            entropy_ring = float(shannon_entropy(ring_vals.ravel(), base=2)) if ring_vals.size else 0.0

            # Sobel gradient magnitude
            grad_mag = sobel(crop.astype(np.float32))
            gradient_mean_ring = grad_mag[ring_mask].mean() if ring_mask.any() else 0.0
            std_gradient_ring  = grad_mag[ring_mask].std()  if ring_mask.any() else 0.0

            # Local Binary Pattern (uniform, P=8, R=1) mean histogram bin value
            if ring_mask.any():
                lbp = local_binary_pattern(crop, P=8, R=1, method="uniform")
                lbp_vals = lbp[ring_mask].astype(int)
                hist, _ = np.histogram(lbp_vals, bins=np.arange(0, 11), density=True)
                lbp_hist_mean = hist.mean()
            else:
                lbp_hist_mean = 0.0

            cell = SampleCell(cell_id=cid, image_id=image_id)
            cell.save_masks({"cytosol": inner_mask, "membrane": ring_mask}, crops_dir)

            x0 = meta["features"].get("x0")
            x1 = meta["features"].get("x1")
            y0 = meta["features"].get("y0")
            y1 = meta["features"].get("y1")

            if None not in (x0, x1, y0, y1):
                cell.set_crop_box(x0, x1, y0, y1)

            cell.set_features(
                cell.features,
                crop_image=f"{cid}.png",
                outer_area=outer_area,
                inner_area=inner_area,
                ring_area=ring_area,
                area_ratio=area_ratio,
                ring_thickness_mean=ring_thickness_mean,
                ring_thickness_std=ring_thickness_std,
                solidity_outer=solidity_outer,
                solidity_inner=solidity_inner,
                eccentricity=eccentricity,
                compactness_outer=compactness_outer,
                compactness_inner=compactness_inner,
                mean_int_ring=mean_ring,
                std_int_ring=std_ring,
                mean_int_inner=mean_inner,
                mean_int_outer=mean_outer,
                contrast_ring_inner=contrast_ring_inner,
                contrast_ring_outer=contrast_ring_outer,
                entropy_ring=float(entropy_ring),
                gradient_mean_ring=gradient_mean_ring,
                std_gradient_ring=std_gradient_ring,
                lbp_hist_mean=lbp_hist_mean
            )


            overlay = green_overlay_with_contours(crop, inner_mask, outer_mask)
            imsave(os.path.join(crops_dir, f"{cid}.png"), overlay)

            records.append(cell.build_feature_record())
            sample_cells.append(cell)

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(output_dir, "features.csv"), index=False)
        SampleCell.save(sample_cells, crops_dir)

        return df

    def score(self, config):
        from traffick_fluo.model.apply import score_features
        score_features(self, config)
