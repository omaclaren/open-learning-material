"""Supplied image-to-court mapping for the ENGSCI 304 TrackID3x3 example.

Requires NumPy. Use:
    from court_mapping import H, transform_points
    court_points = transform_points([[507.61, 412.86]], H)

H maps original 1280 x 720 image coordinates (u, v), in pixels, to court
coordinates (X, Y), in metres. Points are stored as rows of an (n, 2) array.

Provenance
----------
The course calculated H from the four supplied TrackID3x3 Tokoha landmark
correspondences below, using its checked NumPy homography implementation in
learning-notes-scratchwork/instructor-workthroughs/own-scripts/
02_planar_camera_transform.py. The transformation helper is copied from that
implementation. This companion contains the resulting matrix, so fitting is
performed during course preparation rather than when this file is imported.

Dataset: TrackID3x3, indoor basket_S1T4_pre, Yamada et al. (2025).
https://doi.org/10.1145/3728423.3759400
Landmark data: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
The source court coordinates have been converted from centimetres to metres.
"""

from __future__ import annotations

import numpy as np


# Rows: key_1_tokoha, key_6_tokoha, key_12_tokoha, key_9_tokoha.
IMAGE_LANDMARKS_PX = np.array(
    [[509.651, 346.444],
     [1182.068, 423.028],
     [983.043, 678.370],
     [121.543, 430.870]],
    dtype=float,
)
COURT_LANDMARKS_M = np.array(
    [[0.00, 0.00],
     [0.00, 15.05],
     [9.50, 15.05],
     [9.50, 0.00]],
    dtype=float,
)

# Image pixels -> court metres. Retain full precision for calculation;
# round the resulting coordinates only when displaying them.
H = np.array(
    [[0.01002708917422329, -0.08803908415937275, 25.39029644777582],
     [-0.019394496958219598, -0.08915688799021293, 40.77229367213574],
     [-0.00015814557794048556, -0.005042863608550559, 1.0]],
    dtype=float,
)


def _point_array(points, *, name: str) -> np.ndarray:
    """Validate and return an ``(n, 2)`` floating-point point array."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2); received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return array


def transform_points(points, homography) -> np.ndarray:
    """Apply a homography to an ``(n, 2)`` array of points."""
    points = _point_array(points, name="points")
    homography = np.asarray(homography, dtype=float)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("homography must be a finite 3 x 3 matrix")

    homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = homogeneous @ homography.T
    scale = transformed[:, 2]
    if np.any(np.isclose(scale, 0.0)):
        raise ValueError("At least one point maps to infinity under this homography")

    return transformed[:, :2] / scale[:, None]
