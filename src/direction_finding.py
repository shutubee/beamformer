from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from src.adaptive_beamformer import steering_vector_ula


@dataclass(frozen=True)
class DirectionFindingResult:
    """Direction-finding spectrum and detected source angles."""

    scan_angles_deg: NDArray[np.float64]
    spectrum_linear: NDArray[np.float64]
    spectrum_db: NDArray[np.float64]
    detected_angles_deg: NDArray[np.float64]
    detected_peak_levels_db: NDArray[np.float64]
    method: str
    estimated_number_of_sources: int | None


@dataclass(frozen=True)
class ModelOrderResult:
    """Estimated number of narrowband sources."""

    estimated_number_of_sources: int
    criterion_values: NDArray[np.float64]
    candidate_source_counts: NDArray[np.int64]
    eigenvalues: NDArray[np.float64]
    method: str


def calculate_bartlett_spectrum(
    covariance_matrix: NDArray[np.complex128],
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    scan_angles_deg: NDArray[np.float64] | None = None,
    *,
    minimum_db: float = -80.0,
    number_of_sources: int | None = None,
    minimum_peak_distance_deg: float = 2.0,
    minimum_peak_prominence_db: float = 3.0,
) -> DirectionFindingResult:
    """
    Calculate the conventional Bartlett spatial spectrum.

    The Bartlett spectrum is

        P_B(theta) = a(theta)^H R a(theta).

    It is robust and simple, but its angular resolution is limited by
    the physical array aperture.
    """

    covariance = _prepare_covariance_matrix(
        covariance_matrix
    )

    _validate_array_parameters(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
    )

    if covariance.shape != (
        number_of_elements,
        number_of_elements,
    ):
        raise ValueError(
            "covariance_matrix dimensions do not match "
            "number_of_elements."
        )

    scan_angles = _prepare_scan_angles(
        scan_angles_deg
    )

    steering_matrix = _generate_scan_steering_matrix(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        scan_angles_deg=scan_angles,
    )

    covariance_times_steering = (
        covariance @ steering_matrix
    )

    spectrum = np.real(
        np.sum(
            steering_matrix.conj()
            * covariance_times_steering,
            axis=0,
        )
    )

    spectrum = np.maximum(
        spectrum,
        0.0,
    )

    normalized_linear, normalized_db = _normalize_spectrum(
        spectrum,
        minimum_db=minimum_db,
    )

    detected_angles, detected_levels = detect_spectrum_peaks(
        scan_angles_deg=scan_angles,
        spectrum_db=normalized_db,
        number_of_sources=number_of_sources,
        minimum_peak_distance_deg=minimum_peak_distance_deg,
        minimum_peak_prominence_db=minimum_peak_prominence_db,
    )

    return DirectionFindingResult(
        scan_angles_deg=scan_angles,
        spectrum_linear=normalized_linear,
        spectrum_db=normalized_db,
        detected_angles_deg=detected_angles,
        detected_peak_levels_db=detected_levels,
        method="bartlett",
        estimated_number_of_sources=number_of_sources,
    )


def calculate_capon_spectrum(
    covariance_matrix: NDArray[np.complex128],
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    scan_angles_deg: NDArray[np.float64] | None = None,
    *,
    diagonal_loading: float = 1e-3,
    use_pseudoinverse: bool = False,
    minimum_db: float = -80.0,
    number_of_sources: int | None = None,
    minimum_peak_distance_deg: float = 2.0,
    minimum_peak_prominence_db: float = 3.0,
) -> DirectionFindingResult:
    """
    Calculate the Capon or MVDR spatial spectrum.

    The Capon spectrum is

        P_C(theta) =
            1 / [a(theta)^H R^-1 a(theta)].

    It normally gives better resolution than Bartlett, but it depends
    more strongly on covariance quality and numerical conditioning.
    """

    covariance = _prepare_covariance_matrix(
        covariance_matrix
    )

    _validate_array_parameters(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
    )

    if covariance.shape != (
        number_of_elements,
        number_of_elements,
    ):
        raise ValueError(
            "covariance_matrix dimensions do not match "
            "number_of_elements."
        )

    _validate_nonnegative_finite(
        diagonal_loading,
        name="diagonal_loading",
    )

    scan_angles = _prepare_scan_angles(
        scan_angles_deg
    )

    steering_matrix = _generate_scan_steering_matrix(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        scan_angles_deg=scan_angles,
    )

    loaded_covariance = _apply_diagonal_loading(
        covariance,
        diagonal_loading=diagonal_loading,
    )

    if use_pseudoinverse:
        inverse_covariance = np.linalg.pinv(
            loaded_covariance
        )
    else:
        try:
            inverse_covariance = np.linalg.inv(
                loaded_covariance
            )
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "The covariance matrix is singular. Increase diagonal "
                "loading or set use_pseudoinverse=True."
            ) from error

    inverse_times_steering = (
        inverse_covariance
        @ steering_matrix
    )

    denominator = np.real(
        np.sum(
            steering_matrix.conj()
            * inverse_times_steering,
            axis=0,
        )
    )

    numerical_floor = np.finfo(
        np.float64
    ).tiny

    spectrum = 1.0 / np.maximum(
        denominator,
        numerical_floor,
    )

    normalized_linear, normalized_db = _normalize_spectrum(
        spectrum,
        minimum_db=minimum_db,
    )

    detected_angles, detected_levels = detect_spectrum_peaks(
        scan_angles_deg=scan_angles,
        spectrum_db=normalized_db,
        number_of_sources=number_of_sources,
        minimum_peak_distance_deg=minimum_peak_distance_deg,
        minimum_peak_prominence_db=minimum_peak_prominence_db,
    )

    return DirectionFindingResult(
        scan_angles_deg=scan_angles,
        spectrum_linear=normalized_linear,
        spectrum_db=normalized_db,
        detected_angles_deg=detected_angles,
        detected_peak_levels_db=detected_levels,
        method="capon",
        estimated_number_of_sources=number_of_sources,
    )


def calculate_music_spectrum(
    covariance_matrix: NDArray[np.complex128],
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    number_of_sources: int,
    scan_angles_deg: NDArray[np.float64] | None = None,
    *,
    minimum_db: float = -80.0,
    minimum_peak_distance_deg: float = 2.0,
    minimum_peak_prominence_db: float = 3.0,
) -> DirectionFindingResult:
    """
    Calculate the MUSIC direction-of-arrival pseudospectrum.

    MUSIC separates the covariance eigenspace into signal and noise
    subspaces. Its pseudospectrum is

        P_MUSIC(theta) =
            1 / ||E_n^H a(theta)||^2,

    where E_n contains the noise-subspace eigenvectors.
    """

    covariance = _prepare_covariance_matrix(
        covariance_matrix
    )

    _validate_array_parameters(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
    )

    if covariance.shape != (
        number_of_elements,
        number_of_elements,
    ):
        raise ValueError(
            "covariance_matrix dimensions do not match "
            "number_of_elements."
        )

    _validate_source_count(
        number_of_sources=number_of_sources,
        number_of_elements=number_of_elements,
    )

    scan_angles = _prepare_scan_angles(
        scan_angles_deg
    )

    steering_matrix = _generate_scan_steering_matrix(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        scan_angles_deg=scan_angles,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    ascending_indices = np.argsort(
        eigenvalues
    )

    eigenvectors = eigenvectors[
        :,
        ascending_indices,
    ]

    number_of_noise_vectors = (
        number_of_elements
        - number_of_sources
    )

    noise_subspace = eigenvectors[
        :,
        :number_of_noise_vectors,
    ]

    projected_steering = (
        noise_subspace.conj().T
        @ steering_matrix
    )

    denominator = np.sum(
        np.abs(projected_steering) ** 2,
        axis=0,
    )

    numerical_floor = np.finfo(
        np.float64
    ).tiny

    spectrum = 1.0 / np.maximum(
        denominator,
        numerical_floor,
    )

    normalized_linear, normalized_db = _normalize_spectrum(
        spectrum,
        minimum_db=minimum_db,
    )

    detected_angles, detected_levels = detect_spectrum_peaks(
        scan_angles_deg=scan_angles,
        spectrum_db=normalized_db,
        number_of_sources=number_of_sources,
        minimum_peak_distance_deg=minimum_peak_distance_deg,
        minimum_peak_prominence_db=minimum_peak_prominence_db,
    )

    return DirectionFindingResult(
        scan_angles_deg=scan_angles,
        spectrum_linear=normalized_linear,
        spectrum_db=normalized_db,
        detected_angles_deg=detected_angles,
        detected_peak_levels_db=detected_levels,
        method="music",
        estimated_number_of_sources=number_of_sources,
    )


def estimate_source_count_mdl(
    covariance_matrix: NDArray[np.complex128],
    number_of_snapshots: int,
    *,
    maximum_sources: int | None = None,
) -> ModelOrderResult:
    """
    Estimate source count using the minimum description length criterion.

    MDL is usually more conservative than AIC and tends to avoid
    overestimating the number of sources.
    """

    return _estimate_source_count_information_criterion(
        covariance_matrix=covariance_matrix,
        number_of_snapshots=number_of_snapshots,
        maximum_sources=maximum_sources,
        method="mdl",
    )


def estimate_source_count_aic(
    covariance_matrix: NDArray[np.complex128],
    number_of_snapshots: int,
    *,
    maximum_sources: int | None = None,
) -> ModelOrderResult:
    """
    Estimate source count using Akaike's information criterion.

    AIC can resolve weak sources more readily than MDL, but may
    overestimate source count in noisy or limited-snapshot cases.
    """

    return _estimate_source_count_information_criterion(
        covariance_matrix=covariance_matrix,
        number_of_snapshots=number_of_snapshots,
        maximum_sources=maximum_sources,
        method="aic",
    )


def detect_spectrum_peaks(
    scan_angles_deg: NDArray[np.float64],
    spectrum_db: NDArray[np.float64],
    *,
    number_of_sources: int | None = None,
    minimum_peak_distance_deg: float = 2.0,
    minimum_peak_prominence_db: float = 3.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Detect direction-of-arrival peaks in a spatial spectrum.

    When ``number_of_sources`` is supplied, the strongest requested
    number of peaks are returned. Otherwise, every qualifying local
    maximum is returned.
    """

    scan_angles = np.asarray(
        scan_angles_deg,
        dtype=np.float64,
    )

    spectrum = np.asarray(
        spectrum_db,
        dtype=np.float64,
    )

    if scan_angles.ndim != 1 or spectrum.ndim != 1:
        raise ValueError(
            "scan_angles_deg and spectrum_db must be one-dimensional."
        )

    if scan_angles.shape != spectrum.shape:
        raise ValueError(
            "scan_angles_deg and spectrum_db must have equal shape."
        )

    if scan_angles.size < 3:
        raise ValueError(
            "At least three spectrum samples are required."
        )

    if not np.all(
        np.isfinite(scan_angles)
    ) or not np.all(
        np.isfinite(spectrum)
    ):
        raise ValueError(
            "Spectrum inputs contain non-finite values."
        )

    if not np.all(
        np.diff(scan_angles) > 0.0
    ):
        raise ValueError(
            "scan_angles_deg must be strictly increasing."
        )

    _validate_positive_finite(
        minimum_peak_distance_deg,
        name="minimum_peak_distance_deg",
    )
    _validate_nonnegative_finite(
        minimum_peak_prominence_db,
        name="minimum_peak_prominence_db",
    )

    if number_of_sources is not None:
        if isinstance(
            number_of_sources,
            bool,
        ) or not isinstance(
            number_of_sources,
            (int, np.integer),
        ):
            raise TypeError(
                "number_of_sources must be an integer."
            )

        if number_of_sources < 1:
            raise ValueError(
                "number_of_sources must be at least 1."
            )

    angular_step_deg = float(
        np.median(
            np.diff(scan_angles)
        )
    )

    minimum_distance_samples = max(
        1,
        int(
            np.ceil(
                minimum_peak_distance_deg
                / angular_step_deg
            )
        ),
    )

    peak_indices, properties = find_peaks(
        spectrum,
        distance=minimum_distance_samples,
        prominence=minimum_peak_prominence_db,
    )

    if peak_indices.size == 0:
        strongest_index = int(
            np.argmax(spectrum)
        )

        peak_indices = np.array(
            [strongest_index],
            dtype=np.int64,
        )

    peak_levels = spectrum[
        peak_indices
    ]

    descending_order = np.argsort(
        peak_levels
    )[::-1]

    peak_indices = peak_indices[
        descending_order
    ]

    if number_of_sources is not None:
        peak_indices = peak_indices[
            :number_of_sources
        ]

    angular_order = np.argsort(
        scan_angles[
            peak_indices
        ]
    )

    peak_indices = peak_indices[
        angular_order
    ]

    return (
        scan_angles[
            peak_indices
        ].astype(np.float64),
        spectrum[
            peak_indices
        ].astype(np.float64),
    )


def calculate_direction_errors(
    detected_angles_deg: NDArray[np.float64],
    true_angles_deg: NDArray[np.float64],
) -> dict[str, float | NDArray[np.float64]]:
    """
    Match detected and true angles and calculate estimation errors.

    A greedy nearest-angle assignment is used. This is suitable for
    small source sets with reasonably separated directions.
    """

    detected = np.asarray(
        detected_angles_deg,
        dtype=np.float64,
    )

    true_angles = np.asarray(
        true_angles_deg,
        dtype=np.float64,
    )

    if detected.ndim != 1 or true_angles.ndim != 1:
        raise ValueError(
            "Angle arrays must be one-dimensional."
        )

    if detected.size != true_angles.size:
        raise ValueError(
            "Detected and true angle arrays must have equal length."
        )

    if detected.size < 1:
        raise ValueError(
            "At least one source angle is required."
        )

    remaining_detected = list(
        detected
    )

    matched_detected: list[float] = []
    errors: list[float] = []

    for true_angle in true_angles:
        differences = np.abs(
            np.asarray(
                remaining_detected,
                dtype=np.float64,
            )
            - true_angle
        )

        nearest_index = int(
            np.argmin(
                differences
            )
        )

        matched_angle = float(
            remaining_detected.pop(
                nearest_index
            )
        )

        matched_detected.append(
            matched_angle
        )

        errors.append(
            matched_angle
            - float(true_angle)
        )

    error_array = np.asarray(
        errors,
        dtype=np.float64,
    )

    return {
        "matched_detected_angles_deg": np.asarray(
            matched_detected,
            dtype=np.float64,
        ),
        "errors_deg": error_array,
        "mean_error_deg": float(
            np.mean(
                error_array
            )
        ),
        "mean_absolute_error_deg": float(
            np.mean(
                np.abs(
                    error_array
                )
            )
        ),
        "root_mean_square_error_deg": float(
            np.sqrt(
                np.mean(
                    error_array**2
                )
            )
        ),
        "maximum_absolute_error_deg": float(
            np.max(
                np.abs(
                    error_array
                )
            )
        ),
    }


def _estimate_source_count_information_criterion(
    covariance_matrix: NDArray[np.complex128],
    number_of_snapshots: int,
    maximum_sources: int | None,
    method: str,
) -> ModelOrderResult:
    """Shared AIC and MDL source-count estimator."""

    covariance = _prepare_covariance_matrix(
        covariance_matrix
    )

    _validate_positive_integer(
        number_of_snapshots,
        name="number_of_snapshots",
    )

    number_of_elements = covariance.shape[0]

    if maximum_sources is None:
        maximum_sources = (
            number_of_elements - 1
        )
    else:
        if isinstance(
            maximum_sources,
            bool,
        ) or not isinstance(
            maximum_sources,
            (int, np.integer),
        ):
            raise TypeError(
                "maximum_sources must be an integer."
            )

        if not 0 <= maximum_sources < number_of_elements:
            raise ValueError(
                "maximum_sources must lie between 0 and N - 1."
            )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    eigenvalues = np.sort(
        np.real(
            eigenvalues
        )
    )[::-1]

    numerical_floor = np.finfo(
        np.float64
    ).tiny

    eigenvalues = np.maximum(
        eigenvalues,
        numerical_floor,
    )

    candidate_counts = np.arange(
        0,
        maximum_sources + 1,
        dtype=np.int64,
    )

    criterion_values = np.empty(
        candidate_counts.size,
        dtype=np.float64,
    )

    for output_index, source_count in enumerate(
        candidate_counts
    ):
        noise_eigenvalues = eigenvalues[
            source_count:
        ]

        number_of_noise_eigenvalues = (
            number_of_elements
            - source_count
        )

        arithmetic_mean = float(
            np.mean(
                noise_eigenvalues
            )
        )

        geometric_mean = float(
            np.exp(
                np.mean(
                    np.log(
                        noise_eigenvalues
                    )
                )
            )
        )

        ratio = max(
            geometric_mean
            / arithmetic_mean,
            numerical_floor,
        )

        likelihood_term = (
            -number_of_snapshots
            * number_of_noise_eigenvalues
            * np.log(
                ratio
            )
        )

        if method == "mdl":
            penalty = (
                0.5
                * source_count
                * (
                    2
                    * number_of_elements
                    - source_count
                )
                * np.log(
                    number_of_snapshots
                )
            )

            criterion = (
                likelihood_term
                + penalty
            )

        elif method == "aic":
            penalty = (
                source_count
                * (
                    2
                    * number_of_elements
                    - source_count
                )
            )

            criterion = (
                2.0
                * (
                    likelihood_term
                    + penalty
                )
            )

        else:
            raise ValueError(
                "method must be 'mdl' or 'aic'."
            )

        criterion_values[
            output_index
        ] = criterion

    estimated_source_count = int(
        candidate_counts[
            np.argmin(
                criterion_values
            )
        ]
    )

    return ModelOrderResult(
        estimated_number_of_sources=estimated_source_count,
        criterion_values=criterion_values,
        candidate_source_counts=candidate_counts,
        eigenvalues=eigenvalues.astype(
            np.float64
        ),
        method=method,
    )


def _generate_scan_steering_matrix(
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    scan_angles_deg: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Generate one steering vector per scan angle."""

    return np.column_stack(
        [
            steering_vector_ula(
                number_of_elements=number_of_elements,
                frequency_hz=frequency_hz,
                element_spacing_m=element_spacing_m,
                angle_deg=float(
                    angle
                ),
            )
            for angle in scan_angles_deg
        ]
    ).astype(np.complex128)


def _normalize_spectrum(
    spectrum_linear: NDArray[np.float64],
    *,
    minimum_db: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Normalize a nonnegative spectrum to unity and zero dB."""

    spectrum = np.asarray(
        spectrum_linear,
        dtype=np.float64,
    )

    if spectrum.ndim != 1:
        raise ValueError(
            "spectrum_linear must be one-dimensional."
        )

    if not np.all(
        np.isfinite(spectrum)
    ):
        raise ValueError(
            "spectrum_linear contains non-finite values."
        )

    if np.any(
        spectrum < 0.0
    ):
        raise ValueError(
            "spectrum_linear cannot contain negative values."
        )

    if not np.isfinite(
        minimum_db
    ) or minimum_db >= 0.0:
        raise ValueError(
            "minimum_db must be finite and below 0 dB."
        )

    maximum = float(
        np.max(
            spectrum
        )
    )

    if np.isclose(
        maximum,
        0.0,
    ):
        normalized_linear = np.zeros_like(
            spectrum
        )
    else:
        normalized_linear = (
            spectrum
            / maximum
        )

    minimum_linear = 10.0 ** (
        minimum_db / 10.0
    )

    normalized_db = 10.0 * np.log10(
        np.maximum(
            normalized_linear,
            minimum_linear,
        )
    )

    normalized_db = np.maximum(
        normalized_db,
        minimum_db,
    )

    return (
        normalized_linear.astype(
            np.float64
        ),
        normalized_db.astype(
            np.float64
        ),
    )


def _apply_diagonal_loading(
    covariance_matrix: NDArray[np.complex128],
    *,
    diagonal_loading: float,
) -> NDArray[np.complex128]:
    """Apply trace-scaled diagonal loading."""

    if diagonal_loading == 0.0:
        return covariance_matrix.copy()

    number_of_elements = covariance_matrix.shape[0]

    average_power = float(
        np.real(
            np.trace(
                covariance_matrix
            )
        )
        / number_of_elements
    )

    return (
        covariance_matrix
        + diagonal_loading
        * average_power
        * np.eye(
            number_of_elements,
            dtype=np.complex128,
        )
    )


def _prepare_covariance_matrix(
    covariance_matrix: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Validate and Hermitian-symmetrize a covariance matrix."""

    covariance = np.asarray(
        covariance_matrix,
        dtype=np.complex128,
    )

    if covariance.ndim != 2:
        raise ValueError(
            "covariance_matrix must be two-dimensional."
        )

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "covariance_matrix must be square."
        )

    if covariance.shape[0] < 2:
        raise ValueError(
            "covariance_matrix must represent at least two sensors."
        )

    if not np.all(
        np.isfinite(
            covariance.real
        )
    ) or not np.all(
        np.isfinite(
            covariance.imag
        )
    ):
        raise ValueError(
            "covariance_matrix contains non-finite values."
        )

    return np.asarray(
        (
            covariance
            + covariance.conj().T
        )
        / 2.0,
        dtype=np.complex128,
    )


def _prepare_scan_angles(
    scan_angles_deg: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Validate or generate the angular scan grid."""

    if scan_angles_deg is None:
        scan_angles = np.linspace(
            -90.0,
            90.0,
            1801,
            dtype=np.float64,
        )
    else:
        scan_angles = np.asarray(
            scan_angles_deg,
            dtype=np.float64,
        )

    if scan_angles.ndim != 1:
        raise ValueError(
            "scan_angles_deg must be one-dimensional."
        )

    if scan_angles.size < 3:
        raise ValueError(
            "scan_angles_deg must contain at least three values."
        )

    if not np.all(
        np.isfinite(
            scan_angles
        )
    ):
        raise ValueError(
            "scan_angles_deg contains non-finite values."
        )

    if np.any(
        scan_angles < -90.0
    ) or np.any(
        scan_angles > 90.0
    ):
        raise ValueError(
            "scan_angles_deg must lie between -90 and 90 degrees."
        )

    if not np.all(
        np.diff(
            scan_angles
        )
        > 0.0
    ):
        raise ValueError(
            "scan_angles_deg must be strictly increasing."
        )

    return scan_angles


def _validate_array_parameters(
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
) -> None:
    """Validate ULA parameters."""

    _validate_positive_integer(
        number_of_elements,
        name="number_of_elements",
    )

    if number_of_elements < 2:
        raise ValueError(
            "number_of_elements must be at least 2."
        )

    _validate_positive_finite(
        frequency_hz,
        name="frequency_hz",
    )

    _validate_positive_finite(
        element_spacing_m,
        name="element_spacing_m",
    )


def _validate_source_count(
    number_of_sources: int,
    number_of_elements: int,
) -> None:
    """Validate a MUSIC source-count assumption."""

    if isinstance(
        number_of_sources,
        bool,
    ) or not isinstance(
        number_of_sources,
        (int, np.integer),
    ):
        raise TypeError(
            "number_of_sources must be an integer."
        )

    if number_of_sources < 1:
        raise ValueError(
            "number_of_sources must be at least 1."
        )

    if number_of_sources >= number_of_elements:
        raise ValueError(
            "number_of_sources must be smaller than "
            "number_of_elements."
        )


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> None:
    """Validate a positive integer."""

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be positive."
        )


def _validate_positive_finite(
    value: float,
    *,
    name: str,
) -> None:
    """Validate a positive finite scalar."""

    if not np.isfinite(
        value
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if value <= 0.0:
        raise ValueError(
            f"{name} must be positive."
        )


def _validate_nonnegative_finite(
    value: float,
    *,
    name: str,
) -> None:
    """Validate a nonnegative finite scalar."""

    if not np.isfinite(
        value
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if value < 0.0:
        raise ValueError(
            f"{name} cannot be negative."
        )

