# traffick_fluo/stages/segmentation.py

from .base import BaseStage

from cellpose import models
from skimage.io import imread, imsave
from skimage.measure import regionprops
import numpy as np
import pandas as pd
import os

# Project modules
from traffick_fluo.core.crop_utils import crop_with_padding
from traffick_fluo.core.overlay import gray_overlay_with_contour
from traffick_fluo.core.sample import SampleCell
from traffick_fluo.utils.logging import logger
from traffick_fluo.utils.feature_registry import create_feature_registry

# Feature registry
feature, FEATURE_FUNCS = create_feature_registry()

# === Feature functions ===
@feature
def area(region): return region.area

@feature
def mean_intensity(region): return region.mean_intensity

@feature
def intensity_std(region): return np.std(region.intensity_image)

@feature
def solidity(region): return region.solidity

@feature
def eccentricity(region): return region.eccentricity

@feature
def area_perimeter_ratio(region): return region.area / (region.perimeter + 1e-5)


class SegmentationStage(BaseStage):
    ORDER = 1

    FEATURE_FUNCS = FEATURE_FUNCS

    def __init__(self):
        self.model = models.Cellpose(gpu=False, model_type="cyto")

    def _render_crop(self, img, mask, region, pad=10):
        ch0 = img[0]  # Assuming channel 0 is grayscale/membrane
        crop_gray, (y0, x0, y1, x1) = crop_with_padding(ch0, region, pad)
        crop_mask = (mask[y0:y1, x0:x1] == region.label).astype(np.uint8)
        crop_rgb = gray_overlay_with_contour(crop_gray, crop_mask, color=(0, 255, 0))
        return crop_gray, crop_rgb, (y0, x0, y1, x1), crop_mask


    def extract_features(self, image_path, output_dir=None, save=True):
        img = imread(image_path)[1]  # membrane channel only
        masks, *_ = self.model.eval(img, channels=[0, 0], diameter=None)

        props = regionprops(masks, intensity_image=img)
        pad = 10
        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        cells = []
        records = []
        stage_name = self.__class__.__name__.replace("Stage", "").lower()
        score_key = f"{stage_name}_score"

        for i, region in enumerate(props):
            cell_id = str(i)
            try:
                crop_gray, crop_overlay, (y0, x0, y1, x1), crop_mask = self._render_crop(img, masks, region, pad)
            except Exception as e:
                logger.warning(f"Skipping cell {cell_id} due to crop error: {e}")
                continue

            crop_filename = f"{cell_id}.png"
            mask_filename = f"{cell_id}_mask.npy"
            imsave(os.path.join(crops_dir, crop_filename), crop_overlay)
            np.save(os.path.join(crops_dir, mask_filename), crop_mask)

            features = {f.__name__: f(region) for f in self.FEATURE_FUNCS}
            cell.set_crop_box(x0, x1, y0, y1)
            features.update({
                "crop_image": crop_filename,
                "mask_path": mask_filename
            })


            cell = SampleCell(
                cell_id=cell_id,
                image_id=os.path.basename(image_path),
                features=features
            )
            cell.set_scores(**{score_key: 1.0})
            records.append(cell.build_feature_record(score_key=score_key))
            cells.append(cell)

        if save:
            SampleCell.save(cells, crops_dir)
            pd.DataFrame(records).to_csv(os.path.join(output_dir, "features.csv"), index=False)

        logger.info(f"Segmentation: Saved {len(cells)} cells to {output_dir}")
        return pd.DataFrame(records), cells

    def score(self, features_df, model_path):
        # Implemented via model.apply.score_features
        pass