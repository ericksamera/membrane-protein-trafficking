# traffick_fluo/core/crop_utils.py

import numpy as np
from skimage.io import imread
from skimage.measure import _regionprops
import os
from typing import Tuple, Optional, Dict


def extract_crop_bounds(meta: dict) -> Optional[Tuple[int, int, int, int]]:
    """
    Extracts crop bounds from cell metadata.

    Parameters
    ----------
    meta : dict
        Metadata dictionary containing 'features' with keys 'y0', 'y1', 'x0', 'x1'.

    Returns
    -------
    tuple[int, int, int, int] or None
        Crop bounds as (y0, y1, x0, x1), or None if any coordinate is missing.
    """
    y0, x0, y1, x1 = (
        meta["features"].get("y0"),
        meta["features"].get("x0"),
        meta["features"].get("y1"),
        meta["features"].get("x1"),
    )
    if None in (x0, x1, y0, y1):
        return None
    return y0, y1, x0, x1


def crop_image(img: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crops a 2D image using the provided bounds.

    Parameters
    ----------
    img : np.ndarray
        2D image array.
    bounds : tuple[int, int, int, int]
        Crop bounds as (y0, y1, x0, x1).

    Returns
    -------
    np.ndarray
        Cropped region of the image.
    """
    y0, y1, x0, x1 = bounds
    return img[y0:y1, x0:x1]


def load_crop_and_masks(image_path: str, mask_paths: Dict[str, str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load an image and associated masks from disk.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    mask_paths : dict
        Dictionary mapping mask names to their file paths.

    Returns
    -------
    tuple
        - np.ndarray: Loaded image.
        - dict[str, np.ndarray]: Dictionary of loaded masks.
    """
    img = imread(image_path)
    masks = {k: np.load(os.path.join(mask_paths[k])) for k in mask_paths}
    return img, masks


def crop_with_padding(
    image: np.ndarray,
    region: _regionprops.RegionProperties,
    pad: int = 10
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop an image around a region with optional padding.

    Parameters
    ----------
    image : np.ndarray
        Single-channel 2D image array.
    region : skimage.measure._regionprops.RegionProperties
        Region object from skimage.regionprops.
    pad : int, optional
        Padding in pixels (default is 10).

    Returns
    -------
    tuple
        - np.ndarray: Cropped image region.
        - tuple[int, int, int, int]: Absolute crop coordinates (y0, x0, y1, x1).
    """
    minr, minc, maxr, maxc = region.bbox
    minr, minc = max(minr - pad, 0), max(minc - pad, 0)
    maxr, maxc = min(maxr + pad, image.shape[0]), min(maxc + pad, image.shape[1])

    crop = image[minr:maxr, minc:maxc]
    return crop, (minr, minc, maxr, maxc)
