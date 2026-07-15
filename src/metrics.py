"""Radiation-pattern metrics for antenna-array beamforming."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


@dataclass(frozen=True)
class BeamMetrics:
    """Important metrics extracted from a normalized array pattern."""

    requested_steering_angle_deg: float
    actual_peak_angle_deg: float
    steering_error_deg: float
    peak_magnitude_db: float
    half_power_beamwidth_deg: float | None
    first_null_beamwidth_deg: float | None
    peak_sidelobe_level_db: float | None
    sidelobe_angle_deg: float | None
    number_of_major_lobes: int
    grating_lobe_detected: bool
    warnings: tuple[str, ...]


def _validate_pattern(
    angles_deg: NDArray[np.float64],
    magnitude_db: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and standardize input pattern arrays."""

    angles = np.asarray(angles_deg, dtype=np.float64)
    pattern_db = np.asarray(magnitude_db, dtype=np.float64)

    if angles.ndim != 1 or pattern_db.ndim != 1:
        raise ValueError("angles_deg and magnitude_db must be one-dimensional.")

    if angles.size != pattern_db.size:
        raise ValueError(
            "angles_deg and magnitude_db must contain the same number "
            "of values."
        )

    if angles.size < 5:
        raise ValueError("At least five pattern samples are required.")

    if not np.all(np.isfinite(angles)):
        raise ValueError("angles_deg contains non-finite values.")

    if not np.all(np.isfinite(pattern_db)):
        raise ValueError("magnitude_db contains non-finite values.")

    if not np.all(np.diff(angles) > 0.0):
        raise ValueError("angles_deg must be strictly increasing.")

    return angles, pattern_db


def _interpolate_threshold_crossing(
    angle_1: float,
    level_1: float,
    angle_2: float,
    level_2: float,
    threshold_db: float,
) -> float:
    """Linearly interpolate an angle at a specified dB threshold."""

    denominator = level_2 - level_1

    if np.isclose(denominator, 0.0):
        return float((angle_1 + angle_2) / 2.0)

    fraction = (threshold_db - level_1) / denominator

    return float(angle_1 + fraction * (angle_2 - angle_1))


def calculate_half_power_beamwidth(
    angles_deg: NDArray[np.float64],
    magnitude_db: NDArray[np.float64],
    main_peak_index: int,
    threshold_db: float = -3.0,
) -> float | None:
    """
    Calculate half-power beamwidth around the principal beam.

    The calculation identifies the left and right crossings of the
    specified threshold relative to the normalized main-beam peak.
    """

    angles, pattern_db = _validate_pattern(angles_deg, magnitude_db)

    if not 0 <= main_peak_index < angles.size:
        raise IndexError("main_peak_index is outside the pattern array.")

    normalized_db = pattern_db - pattern_db[main_peak_index]

    left_crossing = None
    for index in range(main_peak_index, 0, -1):
        inner_level = normalized_db[index]
        outer_level = normalized_db[index - 1]

        if inner_level >= threshold_db and outer_level < threshold_db:
            left_crossing = _interpolate_threshold_crossing(
                angle_1=angles[index - 1],
                level_1=outer_level,
                angle_2=angles[index],
                level_2=inner_level,
                threshold_db=threshold_db,
            )
            break

    right_crossing = None
    for index in range(main_peak_index, angles.size - 1):
        inner_level = normalized_db[index]
        outer_level = normalized_db[index + 1]

        if inner_level >= threshold_db and outer_level < threshold_db:
            right_crossing = _interpolate_threshold_crossing(
                angle_1=angles[index],
                level_1=inner_level,
                angle_2=angles[index + 1],
                level_2=outer_level,
                threshold_db=threshold_db,
            )
            break

    if left_crossing is None or right_crossing is None:
        return None

    return float(right_crossing - left_crossing)


def find_first_null_indices(
    magnitude_linear: NDArray[np.float64],
    main_peak_index: int,
) -> tuple[int | None, int | None]:
    """
    Find the nearest local minima on either side of the main beam.

    These local minima approximate the first-null locations.
    """

    magnitude = np.asarray(magnitude_linear, dtype=np.float64)

    if magnitude.ndim != 1:
        raise ValueError("magnitude_linear must be one-dimensional.")

    if not 0 <= main_peak_index < magnitude.size:
        raise IndexError("main_peak_index is outside the pattern array.")

    minimum_indices, _ = find_peaks(-magnitude)

    left_candidates = minimum_indices[minimum_indices < main_peak_index]
    right_candidates = minimum_indices[minimum_indices > main_peak_index]

    left_null_index = (
        int(left_candidates[-1]) if left_candidates.size > 0 else None
    )
    right_null_index = (
        int(right_candidates[0]) if right_candidates.size > 0 else None
    )

    return left_null_index, right_null_index


def calculate_peak_sidelobe(
    angles_deg: NDArray[np.float64],
    magnitude_db: NDArray[np.float64],
    left_null_index: int | None,
    right_null_index: int | None,
) -> tuple[float | None, float | None]:
    """
    Find the strongest sidelobe outside the first-null main-beam region.

    Returns
    -------
    tuple
        Peak sidelobe level in dB and its corresponding angle.
    """

    angles, pattern_db = _validate_pattern(angles_deg, magnitude_db)

    excluded = np.zeros(pattern_db.size, dtype=bool)

    if left_null_index is