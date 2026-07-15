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

    if left_null_index is not None and right_null_index is not None:
        excluded[left_null_index : right_null_index + 1] = True
    else:
        peak_index = int(np.argmax(pattern_db))

        approximate_half_width = max(1, pattern_db.size // 100)

        lower = max(0, peak_index - approximate_half_width)
        upper = min(pattern_db.size, peak_index + approximate_half_width + 1)

        excluded[lower:upper] = True

    sidelobe_indices, _ = find_peaks(pattern_db)

    sidelobe_indices = sidelobe_indices[~excluded[sidelobe_indices]]

    if sidelobe_indices.size == 0:
        return None, None

    strongest_index = int(
        sidelobe_indices[np.argmax(pattern_db[sidelobe_indices])]
    )

    normalized_level_db = float(
        pattern_db[strongest_index] - np.max(pattern_db)
    )

    return normalized_level_db, float(angles[strongest_index])


def analyze_beam_pattern(
    angles_deg: NDArray[np.float64],
    magnitude_linear: NDArray[np.float64],
    magnitude_db: NDArray[np.float64],
    requested_steering_angle_deg: float,
    major_lobe_threshold_db: float = -3.0,
    steering_tolerance_deg: float = 1.0,
) -> BeamMetrics:
    """
    Analyze a normalized antenna-array radiation pattern.

    Parameters
    ----------
    angles_deg:
        Observation angles in degrees.

    magnitude_linear:
        Normalized linear magnitude of the array factor.

    magnitude_db:
        Normalized array-factor magnitude in decibels.

    requested_steering_angle_deg:
        Desired beam-steering direction.

    major_lobe_threshold_db:
        Threshold used to identify competing high-level lobes.

    steering_tolerance_deg:
        Maximum acceptable angular difference between requested and
        detected steering direction.
    """

    angles, pattern_db = _validate_pattern(angles_deg, magnitude_db)

    magnitude = np.asarray(magnitude_linear, dtype=np.float64)

    if magnitude.shape != angles.shape:
        raise ValueError(
            "magnitude_linear must have the same shape as angles_deg."
        )

    if np.any(magnitude < 0.0):
        raise ValueError("magnitude_linear cannot contain negative values.")

    main_peak_index = int(np.argmax(pattern_db))
    actual_peak_angle = float(angles[main_peak_index])

    steering_error = actual_peak_angle - requested_steering_angle_deg

    half_power_beamwidth = calculate_half_power_beamwidth(
        angles_deg=angles,
        magnitude_db=pattern_db,
        main_peak_index=main_peak_index,
    )

    left_null_index, right_null_index = find_first_null_indices(
        magnitude_linear=magnitude,
        main_peak_index=main_peak_index,
    )

    if left_null_index is not None and right_null_index is not None:
        first_null_beamwidth = float(
            angles[right_null_index] - angles[left_null_index]
        )
    else:
        first_null_beamwidth = None

    sidelobe_level_db, sidelobe_angle_deg = calculate_peak_sidelobe(
        angles_deg=angles,
        magnitude_db=pattern_db,
        left_null_index=left_null_index,
        right_null_index=right_null_index,
    )

    peak_indices, peak_properties = find_peaks(
        pattern_db,
        height=np.max(pattern_db) + major_lobe_threshold_db,
    )

    number_of_major_lobes = int(peak_indices.size)
    grating_lobe_detected = number_of_major_lobes > 1

    warnings: list[str] = []

    if abs(steering_error) > steering_tolerance_deg:
        warnings.append(
            "Detected beam direction differs from the requested steering "
            f"angle by {abs(steering_error):.2f}°."
        )

    if grating_lobe_detected:
        major_lobe_angles = ", ".join(
            f"{angles[index]:.2f}°" for index in peak_indices
        )
        warnings.append(
            "Multiple high-level lobes were detected at "
            f"{major_lobe_angles}. This may indicate grating lobes."
        )

    if half_power_beamwidth is None:
        warnings.append(
            "The half-power beamwidth could not be determined within "
            "the supplied angular range."
        )

    if first_null_beamwidth is None:
        warnings.append(
            "The first-null beamwidth could not be determined reliably."
        )

    return BeamMetrics(
        requested_steering_angle_deg=float(requested_steering_angle_deg),
        actual_peak_angle_deg=actual_peak_angle,
        steering_error_deg=float(steering_error),
        peak_magnitude_db=float(pattern_db[main_peak_index]),
        half_power_beamwidth_deg=half_power_beamwidth,
        first_null_beamwidth_deg=first_null_beamwidth,
        peak_sidelobe_level_db=sidelobe_level_db,
        sidelobe_angle_deg=sidelobe_angle_deg,
        number_of_major_lobes=number_of_major_lobes,
        grating_lobe_detected=grating_lobe_detected,
        warnings=tuple(warnings),
    )