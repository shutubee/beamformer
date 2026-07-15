"""Core beamforming calculations for uniform linear antenna arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class BeamformerResult:
    """Results returned by the array-factor calculation."""

    angles_deg: NDArray[np.float64]
    array_factor: NDArray[np.complex128]
    magnitude: NDArray[np.float64]
    magnitude_db: NDArray[np.float64]
    wavelength_m: float
    wave_number_rad_m: float


def calculate_ula_array_factor(
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    steering_angle_deg: float = 0.0,
    angles_deg: NDArray[np.float64] | None = None,
    weights: NDArray[np.complex128] | None = None,
    weights_include_steering: bool = False,
    minimum_db: float = -120.0,
) -> BeamformerResult:
    """
    Calculate the normalized array factor of a uniform linear array.

    Two steering workflows are supported.

    Workflow A
    ----------
    Let this function apply the steering phase internally:

        weights_include_steering=False
        steering_angle_deg=<desired angle>

    In this mode, ``weights`` should contain amplitude tapering and any
    non-steering element errors only.

    Workflow B
    ----------
    Supply complete complex weights that already include steering:

        weights_include_steering=True
        steering_angle_deg=0.0

    In this mode, the supplied weights may include steering phase,
    amplitude tapering, phase quantization, calibration errors, and
    element failures.

    Parameters
    ----------
    number_of_elements:
        Number of antenna elements in the uniform linear array.

    frequency_hz:
        Operating frequency in hertz.

    element_spacing_m:
        Centre-to-centre distance between adjacent elements in metres.

    steering_angle_deg:
        Desired beam-steering angle in degrees. This parameter is used
        only when ``weights_include_steering`` is False.

    angles_deg:
        Observation-angle vector in degrees. Defaults to -90° through
        +90° with 0.1° spacing.

    weights:
        Optional complex excitation weights, one per array element.
        Uniform excitation is used when omitted.

    weights_include_steering:
        Set to True when ``weights`` already contain the complete
        steering phase.

    minimum_db:
        Lower numerical floor for the normalized dB pattern.

    Returns
    -------
    BeamformerResult
        Complex array factor, normalized magnitude, normalized dB
        magnitude, wavelength, and wave number.

    Notes
    -----
    The ULA is assumed to lie along the x-axis. The observation angle is
    measured from broadside, giving the spatial phase term

        k d sin(theta).

    When steering is handled internally, the array factor is

        AF(theta) =
            sum_n w_n exp[
                j n k d (
                    sin(theta) - sin(theta_0)
                )
            ].

    When complete steering weights are supplied, the array factor is

        AF(theta) =
            sum_n w_n exp[
                j n k d sin(theta)
            ].
    """

    _validate_inputs(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        steering_angle_deg=steering_angle_deg,
        minimum_db=minimum_db,
    )

    wavelength_m = SPEED_OF_LIGHT_M_S / frequency_hz
    wave_number_rad_m = 2.0 * np.pi / wavelength_m

    if angles_deg is None:
        observation_angles_deg = np.linspace(
            -90.0,
            90.0,
            1801,
            dtype=np.float64,
        )
    else:
        observation_angles_deg = np.asarray(
            angles_deg,
            dtype=np.float64,
        )

    _validate_angles(observation_angles_deg)

    excitation_weights = _prepare_weights(
        number_of_elements=number_of_elements,
        weights=weights,
    )

    observation_angles_rad = np.deg2rad(
        observation_angles_deg
    )

    element_indices = np.arange(
        number_of_elements,
        dtype=np.float64,
    )

    observation_spatial_phase = (
        wave_number_rad_m
        * element_spacing_m
        * np.sin(observation_angles_rad)
    )

    if weights_include_steering:
        phase_difference = observation_spatial_phase
    else:
        steering_angle_rad = np.deg2rad(
            steering_angle_deg
        )

        steering_spatial_phase = (
            wave_number_rad_m
            * element_spacing_m
            * np.sin(steering_angle_rad)
        )

        phase_difference = (
            observation_spatial_phase
            - steering_spatial_phase
        )

    steering_matrix = np.exp(
        1j
        * np.outer(
            element_indices,
            phase_difference,
        )
    )

    array_factor = excitation_weights @ steering_matrix

    magnitude = np.abs(array_factor).astype(
        np.float64
    )

    maximum_magnitude = float(np.max(magnitude))

    if not np.isfinite(maximum_magnitude):
        raise FloatingPointError(
            "The calculated array factor contains non-finite values."
        )

    if np.isclose(maximum_magnitude, 0.0):
        normalized_magnitude = np.zeros_like(
            magnitude,
            dtype=np.float64,
        )
    else:
        normalized_magnitude = (
            magnitude / maximum_magnitude
        )

    minimum_linear = 10.0 ** (minimum_db / 20.0)

    normalized_magnitude_db = (
        20.0
        * np.log10(
            np.maximum(
                normalized_magnitude,
                minimum_linear,
            )
        )
    )

    normalized_magnitude_db = np.maximum(
        normalized_magnitude_db,
        minimum_db,
    )

    return BeamformerResult(
        angles_deg=observation_angles_deg,
        array_factor=np.asarray(
            array_factor,
            dtype=np.complex128,
        ),
        magnitude=np.asarray(
            normalized_magnitude,
            dtype=np.float64,
        ),
        magnitude_db=np.asarray(
            normalized_magnitude_db,
            dtype=np.float64,
        ),
        wavelength_m=float(wavelength_m),
        wave_number_rad_m=float(
            wave_number_rad_m
        ),
    )


def _validate_inputs(
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    steering_angle_deg: float,
    minimum_db: float,
) -> None:
    """Validate scalar beamformer inputs."""

    if isinstance(number_of_elements, bool):
        raise TypeError(
            "number_of_elements must be an integer."
        )

    if not isinstance(
        number_of_elements,
        (int, np.integer),
    ):
        raise TypeError(
            "number_of_elements must be an integer."
        )

    if number_of_elements < 2:
        raise ValueError(
            "number_of_elements must be at least 2."
        )

    if not np.isfinite(frequency_hz):
        raise ValueError(
            "frequency_hz must be finite."
        )

    if frequency_hz <= 0.0:
        raise ValueError(
            "frequency_hz must be positive."
        )

    if not np.isfinite(element_spacing_m):
        raise ValueError(
            "element_spacing_m must be finite."
        )

    if element_spacing_m <= 0.0:
        raise ValueError(
            "element_spacing_m must be positive."
        )

    if not np.isfinite(steering_angle_deg):
        raise ValueError(
            "steering_angle_deg must be finite."
        )

    if not -90.0 <= steering_angle_deg <= 90.0:
        raise ValueError(
            "steering_angle_deg must lie between "
            "-90 and 90 degrees."
        )

    if not np.isfinite(minimum_db):
        raise ValueError(
            "minimum_db must be finite."
        )

    if minimum_db >= 0.0:
        raise ValueError(
            "minimum_db must be below 0 dB."
        )


def _validate_angles(
    angles_deg: NDArray[np.float64],
) -> None:
    """Validate the observation-angle vector."""

    if angles_deg.ndim != 1:
        raise ValueError(
            "angles_deg must be one-dimensional."
        )

    if angles_deg.size < 2:
        raise ValueError(
            "angles_deg must contain at least two values."
        )

    if not np.all(np.isfinite(angles_deg)):
        raise ValueError(
            "angles_deg contains non-finite values."
        )

    if np.any(angles_deg < -90.0) or np.any(
        angles_deg > 90.0
    ):
        raise ValueError(
            "All observation angles must lie between "
            "-90 and 90 degrees."
        )

    if not np.all(np.diff(angles_deg) > 0.0):
        raise ValueError(
            "angles_deg must be strictly increasing."
        )


def _prepare_weights(
    number_of_elements: int,
    weights: NDArray[np.complex128] | None,
) -> NDArray[np.complex128]:
    """Validate and prepare element excitation weights."""

    if weights is None:
        excitation_weights = np.ones(
            number_of_elements,
            dtype=np.complex128,
        )
    else:
        excitation_weights = np.asarray(
            weights,
            dtype=np.complex128,
        )

    if excitation_weights.ndim != 1:
        raise ValueError(
            "weights must be one-dimensional."
        )

    expected_shape = (number_of_elements,)

    if excitation_weights.shape != expected_shape:
        raise ValueError(
            f"weights must have shape {expected_shape}, "
            f"but received {excitation_weights.shape}."
        )

    if not np.all(
        np.isfinite(excitation_weights.real)
    ) or not np.all(
        np.isfinite(excitation_weights.imag)
    ):
        raise ValueError(
            "weights contains non-finite values."
        )

    if np.allclose(
        np.abs(excitation_weights),
        0.0,
    ):
        raise ValueError(
            "weights cannot contain only zeros."
        )

    return excitation_weights