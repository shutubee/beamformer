"""Core beamforming calculations for uniform linear antenna arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BeamformerResult:
    """Results returned by the array-factor calculation."""

    angles_deg: NDArray[np.float64]
    array_factor: NDArray[np.complex128]
    magnitude: NDArray[np.float64]
    magnitude_db: NDArray[np.float64]


def calculate_ula_array_factor(
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    steering_angle_deg: float,
    angles_deg: NDArray[np.float64] | None = None,
    weights: NDArray[np.complex128] | None = None,
) -> BeamformerResult:
    """
    Calculate the normalized array factor of a uniform linear array.

    Parameters
    ----------
    number_of_elements:
        Number of antenna elements in the array.

    frequency_hz:
        Operating frequency in hertz.

    element_spacing_m:
        Distance between adjacent antenna elements in metres.

    steering_angle_deg:
        Desired beam-steering angle in degrees.

    angles_deg:
        Observation-angle vector. Defaults to -90° to +90°.

    weights:
        Complex excitation weight for each antenna element.
        Defaults to uniform excitation.

    Returns
    -------
    BeamformerResult
        Complex array factor and normalized linear and dB magnitudes.
    """

    if number_of_elements < 2:
        raise ValueError("number_of_elements must be at least 2.")

    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive.")

    if element_spacing_m <= 0:
        raise ValueError("element_spacing_m must be positive.")

    if not -90.0 <= steering_angle_deg <= 90.0:
        raise ValueError("steering_angle_deg must lie between -90 and 90.")

    speed_of_light = 299_792_458.0
    wavelength_m = speed_of_light / frequency_hz
    wave_number = 2.0 * np.pi / wavelength_m

    if angles_deg is None:
        angles_deg = np.linspace(-90.0, 90.0, 1801)

    angles_deg = np.asarray(angles_deg, dtype=np.float64)

    if weights is None:
        weights = np.ones(number_of_elements, dtype=np.complex128)
    else:
        weights = np.asarray(weights, dtype=np.complex128)

    if weights.shape != (number_of_elements,):
        raise ValueError(
            f"weights must have shape ({number_of_elements},)."
        )

    observation_angles_rad = np.deg2rad(angles_deg)
    steering_angle_rad = np.deg2rad(steering_angle_deg)

    element_indices = np.arange(number_of_elements)

    phase_difference = (
        wave_number
        * element_spacing_m
        * (
            np.sin(observation_angles_rad)
            - np.sin(steering_angle_rad)
        )
    )

    steering_matrix = np.exp(
        1j * np.outer(element_indices, phase_difference)
    )

    array_factor = weights @ steering_matrix

    magnitude = np.abs(array_factor)

    maximum = np.max(magnitude)
    if maximum > 0:
        magnitude = magnitude / maximum

    minimum_linear_value = 1e-12
    magnitude_db = 20.0 * np.log10(
        np.maximum(magnitude, minimum_linear_value)
    )

    return BeamformerResult(
        angles_deg=angles_deg,
        array_factor=array_factor,
        magnitude=magnitude,
        magnitude_db=magnitude_db,
    )