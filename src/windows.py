"""Amplitude-tapering windows for antenna-array beamforming."""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import chebwin, taylor


class WindowType(str, Enum):
    """Supported antenna-array amplitude windows."""

    UNIFORM = "Uniform"
    HANN = "Hann"
    HAMMING = "Hamming"
    BLACKMAN = "Blackman"
    TAYLOR = "Taylor"
    DOLPH_CHEBYSHEV = "Dolph-Chebyshev"


def _validate_number_of_elements(number_of_elements: int) -> None:
    """Validate the antenna-array element count."""

    if not isinstance(number_of_elements, int):
        raise TypeError("number_of_elements must be an integer.")

    if number_of_elements < 2:
        raise ValueError("number_of_elements must be at least 2.")


def normalize_weights(
    weights: NDArray[np.float64] | NDArray[np.complex128],
    mode: str = "peak",
) -> NDArray[np.complex128]:
    """
    Normalize antenna-element excitation weights.

    Parameters
    ----------
    weights:
        Real or complex excitation weights.

    mode:
        Normalization method:

        - ``peak``: maximum magnitude becomes 1.
        - ``sum``: sum of magnitudes becomes 1.
        - ``power``: total squared magnitude becomes 1.

    Returns
    -------
    numpy.ndarray
        Normalized complex excitation weights.
    """

    array = np.asarray(weights, dtype=np.complex128)

    if array.ndim != 1:
        raise ValueError("weights must be one-dimensional.")

    if array.size < 2:
        raise ValueError("weights must contain at least two values.")

    if not np.all(np.isfinite(array)):
        raise ValueError("weights contains non-finite values.")

    magnitude = np.abs(array)

    if np.allclose(magnitude, 0.0):
        raise ValueError("weights cannot contain only zeros.")

    normalized_mode = mode.strip().lower()

    if normalized_mode == "peak":
        scale = np.max(magnitude)

    elif normalized_mode == "sum":
        scale = np.sum(magnitude)

    elif normalized_mode == "power":
        scale = np.sqrt(np.sum(magnitude**2))

    else:
        raise ValueError(
            "mode must be one of: 'peak', 'sum', or 'power'."
        )

    return array / scale


def generate_uniform_window(
    number_of_elements: int,
) -> NDArray[np.float64]:
    """Generate uniform element amplitudes."""

    _validate_number_of_elements(number_of_elements)

    return np.ones(number_of_elements, dtype=np.float64)


def generate_hann_window(
    number_of_elements: int,
) -> NDArray[np.float64]:
    """Generate a symmetric Hann amplitude taper."""

    _validate_number_of_elements(number_of_elements)

    return np.hanning(number_of_elements).astype(np.float64)


def generate_hamming_window(
    number_of_elements: int,
) -> NDArray[np.float64]:
    """Generate a symmetric Hamming amplitude taper."""

    _validate_number_of_elements(number_of_elements)

    return np.hamming(number_of_elements).astype(np.float64)


def generate_blackman_window(
    number_of_elements: int,
) -> NDArray[np.float64]:
    """Generate a symmetric Blackman amplitude taper."""

    _validate_number_of_elements(number_of_elements)

    return np.blackman(number_of_elements).astype(np.float64)


def generate_taylor_window(
    number_of_elements: int,
    number_of_near_sidelobes: int = 4,
    sidelobe_level_db: float = 30.0,
) -> NDArray[np.float64]:
    """
    Generate a Taylor amplitude taper.

    Parameters
    ----------
    number_of_elements:
        Number of antenna elements.

    number_of_near_sidelobes:
        Number of nearly constant-level sidelobes adjacent to the
        main beam.

    sidelobe_level_db:
        Desired positive sidelobe suppression in decibels.

        For example, ``30`` targets sidelobes near -30 dB.
    """

    _validate_number_of_elements(number_of_elements)

    if number_of_near_sidelobes < 2:
        raise ValueError(
            "number_of_near_sidelobes must be at least 2."
        )

    if number_of_near_sidelobes > number_of_elements:
        raise ValueError(
            "number_of_near_sidelobes cannot exceed "
            "number_of_elements."
        )

    if sidelobe_level_db <= 0.0:
        raise ValueError("sidelobe_level_db must be positive.")

    return np.asarray(
        taylor(
            M=number_of_elements,
            nbar=number_of_near_sidelobes,
            sll=sidelobe_level_db,
            norm=True,
            sym=True,
        ),
        dtype=np.float64,
    )


def generate_dolph_chebyshev_window(
    number_of_elements: int,
    attenuation_db: float = 40.0,
) -> NDArray[np.float64]:
    """
    Generate a Dolph-Chebyshev amplitude taper.

    Parameters
    ----------
    number_of_elements:
        Number of antenna elements.

    attenuation_db:
        Positive sidelobe attenuation in decibels.

        Values below approximately 45 dB can produce non-monotonic
        equivalent-noise bandwidth in SciPy's Chebyshev implementation.
    """

    _validate_number_of_elements(number_of_elements)

    if attenuation_db <= 0.0:
        raise ValueError("attenuation_db must be positive.")

    return np.asarray(
        chebwin(
            M=number_of_elements,
            at=attenuation_db,
            sym=True,
        ),
        dtype=np.float64,
    )


def generate_window(
    number_of_elements: int,
    window_type: WindowType | str = WindowType.UNIFORM,
    *,
    normalization: str = "peak",
    taylor_nbar: int = 4,
    sidelobe_level_db: float = 30.0,
    chebyshev_attenuation_db: float = 40.0,
) -> NDArray[np.complex128]:
    """
    Generate normalized antenna-element amplitude weights.

    Parameters
    ----------
    number_of_elements:
        Number of array elements.

    window_type:
        Window type as a ``WindowType`` value or supported string.

    normalization:
        Weight normalization mode: ``peak``, ``sum``, or ``power``.

    taylor_nbar:
        Number of nearly constant Taylor sidelobes.

    sidelobe_level_db:
        Taylor sidelobe suppression target.

    chebyshev_attenuation_db:
        Dolph-Chebyshev sidelobe attenuation target.

    Returns
    -------
    numpy.ndarray
        Normalized complex weights with zero phase.
    """

    _validate_number_of_elements(number_of_elements)

    if isinstance(window_type, WindowType):
        selected_window = window_type
    else:
        normalized_name = str(window_type).strip().lower()

        aliases = {
            "uniform": WindowType.UNIFORM,
            "rectangular": WindowType.UNIFORM,
            "hann": WindowType.HANN,
            "hanning": WindowType.HANN,
            "hamming": WindowType.HAMMING,
            "blackman": WindowType.BLACKMAN,
            "taylor": WindowType.TAYLOR,
            "dolph-chebyshev": WindowType.DOLPH_CHEBYSHEV,
            "dolph chebyshev": WindowType.DOLPH_CHEBYSHEV,
            "chebyshev": WindowType.DOLPH_CHEBYSHEV,
            "chebwin": WindowType.DOLPH_CHEBYSHEV,
        }

        try:
            selected_window = aliases[normalized_name]
        except KeyError as error:
            supported = ", ".join(item.value for item in WindowType)
            raise ValueError(
                f"Unsupported window type '{window_type}'. "
                f"Supported values: {supported}."
            ) from error

    generators = {
        WindowType.UNIFORM: lambda: generate_uniform_window(
            number_of_elements
        ),
        WindowType.HANN: lambda: generate_hann_window(
            number_of_elements
        ),
        WindowType.HAMMING: lambda: generate_hamming_window(
            number_of_elements
        ),
        WindowType.BLACKMAN: lambda: generate_blackman_window(
            number_of_elements
        ),
        WindowType.TAYLOR: lambda: generate_taylor_window(
            number_of_elements=number_of_elements,
            number_of_near_sidelobes=taylor_nbar,
            sidelobe_level_db=sidelobe_level_db,
        ),
        WindowType.DOLPH_CHEBYSHEV: lambda: (
            generate_dolph_chebyshev_window(
                number_of_elements=number_of_elements,
                attenuation_db=chebyshev_attenuation_db,
            )
        ),
    }

    weights = generators[selected_window]()

    return normalize_weights(weights, mode=normalization)