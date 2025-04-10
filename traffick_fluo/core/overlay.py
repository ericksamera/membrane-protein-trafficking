# traffick_fluo/core/overlay.py

import numpy as np
import cv2
from typing import Tuple


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalize an image to uint8 [0, 255] range.

    Parameters
    ----------
    img : np.ndarray
        Input image (float or int).

    Returns
    -------
    np.ndarray
        Normalized uint8 image.
    """
    norm = ((img - img.min()) / (img.ptp() + 1e-5) * 255).astype(np.uint8)
    return norm


def overlay_contours(
    base_img: np.ndarray,
    contours: list,
    color: Tuple[int, int, int],
    thickness: int = 1
) -> np.ndarray:
    """
    Draw contours on a base RGB image.

    Parameters
    ----------
    base_img : np.ndarray
        RGB image to draw on.
    contours : list
        Contours as returned by cv2.findContours.
    color : tuple[int, int, int]
        RGB color for contours.
    thickness : int, optional
        Thickness of contour lines.

    Returns
    -------
    np.ndarray
        Image with contours drawn.
    """
    return cv2.drawContours(base_img, contours, -1, color, thickness)


def gray_overlay_with_contour(
    crop_gray: np.ndarray,
    crop_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Overlay a binary mask as a contour on a grayscale image.

    Parameters
    ----------
    crop_gray : np.ndarray
        Grayscale image.
    crop_mask : np.ndarray
        Binary mask.
    color : tuple[int, int, int], optional
        RGB color of the contour.

    Returns
    -------
    np.ndarray
        RGB overlay image with contour.
    """
    norm = normalize_to_uint8(crop_gray)
    rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    cnts, _ = cv2.findContours(crop_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, cnts, -1, color, 1)
    return rgb


def green_overlay_with_contours(
    crop: np.ndarray,
    cyto_mask: np.ndarray,
    ring_mask: np.ndarray
) -> np.ndarray:
    """
    Overlay membrane and cytosol masks on a crop using color contours.

    Parameters
    ----------
    crop : np.ndarray
        Grayscale image.
    cyto_mask : np.ndarray
        Cytosol mask (shown in red).
    ring_mask : np.ndarray
        Membrane mask (shown in blue).

    Returns
    -------
    np.ndarray
        RGB overlay image.
    """
    norm = normalize_to_uint8(crop)
    rgb = np.zeros((*norm.shape, 3), dtype=np.uint8)
    rgb[..., 1] = norm  # green channel

    cyto_cnts, _ = cv2.findContours(cyto_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ring_cnts, _ = cv2.findContours(ring_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay_contours(rgb, ring_cnts, (255, 0, 0))  # blue for membrane
    overlay_contours(rgb, cyto_cnts, (0, 0, 255))  # red for cytosol
    return rgb


def magenta_overlay(crop: np.ndarray) -> np.ndarray:
    """
    Create a magenta overlay (R+B) from a single-channel image.

    Parameters
    ----------
    crop : np.ndarray
        Grayscale image.

    Returns
    -------
    np.ndarray
        RGB image with magenta tone.
    """
    norm = normalize_to_uint8(crop)
    rgb = np.zeros((*norm.shape, 3), dtype=np.uint8)
    rgb[..., 0] = norm  # red
    rgb[..., 2] = norm  # blue
    return rgb
