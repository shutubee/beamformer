"""Steering-weight generation for phased antenna arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class SteeringWeights:
    """Element-level steering-weight information."""

    element_indices: NDArray[np.int64]
    ideal_phase_rad: NDArray[np.float64]
    applied_phase_rad: NDArray[np.float64]
    amplitude: NDArray[np.float64]
    complex_weights: NDArray[np.complex128]
    phase_error_rad: NDArray[np.float64]
    quantization_bits: int | None


def _validate_inputs(
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    steering_angle_deg: float,
) -> None:
    """Validate common phased-array steering inputs."""

    if not isinstance(number_of_elements, int):
        raise TypeError("number_of_elements must be an integer.")

    if number_of_elements < 2:
        raise ValueError("number_of_elements must be at least 2.")

    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")

    if element_spacing_m <= 0.0:
        raise ValueError("element_spacing_m must be positive.")

    if not -90.0 <= steering_angle_deg <= 90.0:
        raise ValueError(
            "steering_angle_deg must lie between -90 and 90 degrees."
        )


def wrap_phase_rad(
    phase_rad: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Wrap phase values to the interval [-π, π)."""

    phase = np.asarray(phase_rad, dtype=np.float64)

    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def calculate_progressive_phase_rad(
    frequency_hz: float,
    element_spacing_m: float,
    steering_angle_deg: float,
) -> float:
    """
    Calculate the progressive phase shift between adjacent elements.

    The sign convention is consistent with the array-factor expression

        exp[j n k d (sin(theta) - sin(theta_0))].

    Therefore, the excitation phase for element n is

        -n k d sin(theta_0).
    """

    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")

    if element_spacing_m <= 0.0:
        raise ValueError("element_spacing_m must be positive.")

    if not -90.0 <= steering_angle_deg <= 90.0:
        raise ValueError(
            "steering_angle_deg must lie between -90 and 90 degrees."
        )

    wavelength_m = SPEED_OF_LIGHT_M_S / frequency_hz
    wave_number_rad_m = 2.0 * np.pi / wavelength_m
    steering_angle_rad = np.deg2rad(steering_angle_deg)

    return float(
        -wave_number_rad_m
        * element_spacing_m
        * np.sin(steering_angle_rad)
    )


def generate_ideal_steering_phase(
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    steering_angle_deg: float,
    *,
    wrap_phase: bool = True,
) -> NDArray[np.float64]:
    """Generate ideal per-element phase shifts for a ULA."""

    _validate_inputs(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        steering_angle_deg=steering_angle_deg,
    )

    progressive_phase_rad = calculate_progressive_phase_rad(
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        steering_angle_deg=steering_angle_deg,
    )

    element_indices = np.arange(
        number_of_elements,
        dtype=np.float64,
    )

    phases_rad = element_indices * progressive_phase_rad

    if wrap_phase:
        phases_rad = wrap_phase_rad(phases_rad)

    return np.asarray(phases_rad, dtype=np.float64)


def quantize_phase_rad(
    phase_rad: NDArray[np.float64],
    number_of_bits: int,
) -> NDArray[np.float64]:
    """
    Quantize phases using a finite-resolution phase shifter.

    A B-bit phase shifter has 2**B discrete phase states over 2π.
    """

    phase = np.asarray(phase_rad, dtype=np.float64)

    if phase.ndim != 1:
        raise ValueError("phase_rad must be one-dimensional.")

    if not isinstance(number_of_bits, int):
        raise TypeError("number_of_bits must be an integer.")

    if number_of_bits < 1:
        raise ValueError("number_of_bits must be at least 1.")

    number_of_states = 2**number_of_bits
    phase_step_rad = 2.0 * np.pi / number_of_states

    phase_0_to_2pi = np.mod(phase, 2.0 * np.pi)

    quantized = (
        np.round(phase_0_to_2pi / phase_step_rad)
        * phase_step_rad
    )

    return wrap_phase_rad(quantized)


def generate_phase_errors(
    number_of_elements: int,
    standard_deviation_deg: float,
    *,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate independent Gaussian phase errors."""

    if number_of_elements < 2:
        raise ValueError("number_of_elements must be at least 2.")

    if standard_deviation_deg < 0.0:
        raise ValueError(
            "standard_deviation_deg cannot be negative."
        )

    if standard_deviation_deg == 0.0:
        return np.zeros(number_of_elements, dtype=np.float64)

    random_generator = np.random.default_rng(seed)

    phase_errors_deg = random_generator.normal(
        loc=0.0,
        scale=standard_deviation_deg,
        size=number_of_elements,
    )

    return np.deg2rad(phase_errors_deg).astype(np.float64)


def generate_steering_weights(
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    steering_angle_deg: float,
    *,
    amplitude_weights: NDArray[np.float64] | None = None,
    quantization_bits: int | None = None,
    phase_error_std_deg: float = 0.0,
    phase_errors_rad: NDArray[np.float64] | None = None,
    seed: int | None = None,
    normalize_amplitude: bool = True,
) -> SteeringWeights:
    """
    Generate practical complex weights for a uniform linear array.

    Parameters
    ----------
    number_of_elements:
        Number of array elements.

    frequency_hz:
        Operating frequency in hertz.

    element_spacing_m:
        Adjacent-element spacing in metres.

    steering_angle_deg:
        Requested steering angle.

    amplitude_weights:
        Optional real-valued amplitude taper. Uniform amplitudes are
        used when omitted.

    quantization_bits:
        Optional finite phase-shifter resolution.

    phase_error_std_deg:
        Standard deviation of random phase error in degrees.

    phase_errors_rad:
        Optional explicit per-element phase-error vector. When supplied,
        this takes precedence over ``phase_error_std_deg``.

    seed:
        Random seed used for reproducible phase errors.

    normalize_amplitude:
        Normalize the largest amplitude to one.
    """

    _validate_inputs(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        steering_angle_deg=steering_angle_deg,
    )

    ideal_phase_rad = generate_ideal_steering_phase(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        steering_angle_deg=steering_angle_deg,
        wrap_phase=True,
    )

    if amplitude_weights is None:
        amplitude = np.ones(
            number_of_elements,
            dtype=np.float64,
        )
    else:
        amplitude = np.asarray(
            amplitude_weights,
            dtype=np.float64,
        )

    if amplitude.shape != (number_of_elements,):
        raise ValueError(
            f"amplitude_weights must have shape "
            f"({number_of_elements},)."
        )

    if not np.all(np.isfinite(amplitude)):
        raise ValueError(
            "amplitude_weights contains non-finite values."
        )

    if np.any(amplitude < 0.0):
        raise ValueError(
            "amplitude_weights cannot contain negative values."
        )

    if np.allclose(amplitude, 0.0):
        raise ValueError(
            "amplitude_weights cannot contain only zeros."
        )

    if normalize_amplitude:
        amplitude = amplitude / np.max(amplitude)

    applied_phase_rad = ideal_phase_rad.copy()

    if quantization_bits is not None:
        applied_phase_rad = quantize_phase_rad(
            phase_rad=applied_phase_rad,
            number_of_bits=quantization_bits,
        )

    if phase_errors_rad is not None:
        phase_error = np.asarray(
            phase_errors_rad,
            dtype=np.float64,
        )

        if phase_error.shape != (number_of_elements,):
            raise ValueError(
                f"phase_errors_rad must have shape "
                f"({number_of_elements},)."
            )

        if not np.all(np.isfinite(phase_error)):
            raise ValueError(
                "phase_errors_rad contains non-finite values."
            )
    else:
        phase_error = generate_phase_errors(
            number_of_elements=number_of_elements,
            standard_deviation_deg=phase_error_std_deg,
            seed=seed,
        )

    applied_phase_rad = wrap_phase_rad(
        applied_phase_rad + phase_error
    )

    complex_weights = amplitude * np.exp(
        1j * applied_phase_rad
    )

    return SteeringWeights(
        element_indices=np.arange(
            number_of_elements,
            dtype=np.int64,
        ),
        ideal_phase_rad=ideal_phase_rad,
        applied_phase_rad=applied_phase_rad,
        amplitude=amplitude,
        complex_weights=np.asarray(
            complex_weights,
            dtype=np.complex128,
        ),
        phase_error_rad=phase_error,
        quantization_bits=quantization_bits,
    )