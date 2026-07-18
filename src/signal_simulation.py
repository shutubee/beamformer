"""Narrowband signal and receiver-snapshot simulation utilities."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from src.adaptive_beamformer import steering_vector_ula
@dataclass(frozen=True)
class SourceDefinition:
    """Definition of one narrowband far-field source."""
    angle_deg: float
    power_db: float
    label: str
    signal_type: str = "complex_gaussian"
    correlation_group: str | None = None
    correlation_coefficient: complex = 0.0 + 0.0j
@dataclass(frozen=True)
class SnapshotSimulationResult:
    """Output of a narrowband array snapshot simulation."""
    snapshots: NDArray[np.complex128]
    clean_snapshots: NDArray[np.complex128]
    noise_snapshots: NDArray[np.complex128]
    source_signals: NDArray[np.complex128]
    steering_matrix: NDArray[np.complex128]
    source_angles_deg: NDArray[np.float64]
    source_powers_linear: NDArray[np.float64]
    source_labels: tuple[str, ...]
    noise_power_linear: float
    sample_covariance: NDArray[np.complex128]
    theoretical_covariance: NDArray[np.complex128]
    number_of_elements: int
    number_of_snapshots: int
def db_to_linear_power(
    power_db: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Convert a power quantity from decibels to linear units."""
    values = np.asarray(
        power_db,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "power_db contains non-finite values."
        )
    return np.asarray(
        10.0 ** (values / 10.0),
        dtype=np.float64,
    )
def linear_power_to_db(
    power_linear: NDArray[np.float64] | float,
    *,
    minimum_db: float = -300.0,
) -> NDArray[np.float64]:
    """Convert a nonnegative linear power quantity to decibels."""
    values = np.asarray(
        power_linear,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "power_linear contains non-finite values."
        )
    if np.any(values < 0.0):
        raise ValueError(
            "power_linear cannot contain negative values."
        )
    if not np.isfinite(minimum_db) or minimum_db >= 0.0:
        raise ValueError(
            "minimum_db must be finite and below 0 dB."
        )
    minimum_linear = 10.0 ** (
        minimum_db / 10.0
    )
    return np.maximum(
        10.0
        * np.log10(
            np.maximum(
                values,
                minimum_linear,
            )
        ),
        minimum_db,
    ).astype(np.float64)
def generate_complex_gaussian_signal(
    number_of_snapshots: int,
    *,
    power_linear: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    Generate a circularly symmetric complex Gaussian signal.
    The generated process has approximately
        E[|s[k]|²] = power_linear.
    """
    _validate_positive_integer(
        number_of_snapshots,
        name="number_of_snapshots",
    )
    _validate_nonnegative_finite(
        power_linear,
        name="power_linear",
    )
    if power_linear == 0.0:
        return np.zeros(
            number_of_snapshots,
            dtype=np.complex128,
        )
    random_generator = np.random.default_rng(
        seed
    )
    signal = (
        random_generator.normal(
            loc=0.0,
            scale=1.0,
            size=number_of_snapshots,
        )
        + 1j
        * random_generator.normal(
            loc=0.0,
            scale=1.0,
            size=number_of_snapshots,
        )
    ) / np.sqrt(2.0)
    signal *= np.sqrt(power_linear)
    return np.asarray(
        signal,
        dtype=np.complex128,
    )
def generate_complex_tone(
    number_of_snapshots: int,
    normalized_frequency: float,
    *,
    power_linear: float = 1.0,
    initial_phase_rad: float = 0.0,
) -> NDArray[np.complex128]:
    """
    Generate a sampled complex sinusoid.
    Parameters
    ----------
    number_of_snapshots:
        Number of output samples.
    normalized_frequency:
        Cycles per snapshot. It should normally lie between -0.5 and
        +0.5.
    power_linear:
        Signal power. The tone magnitude is ``sqrt(power_linear)``.
    initial_phase_rad:
        Starting phase in radians.
    """
    _validate_positive_integer(
        number_of_snapshots,
        name="number_of_snapshots",
    )
    _validate_nonnegative_finite(
        power_linear,
        name="power_linear",
    )
    if not np.isfinite(
        normalized_frequency
    ):
        raise ValueError(
            "normalized_frequency must be finite."
        )
    if not -0.5 <= normalized_frequency <= 0.5:
        raise ValueError(
            "normalized_frequency must lie between -0.5 and 0.5."
        )
    if not np.isfinite(initial_phase_rad):
        raise ValueError(
            "initial_phase_rad must be finite."
        )
    sample_indices = np.arange(
        number_of_snapshots,
        dtype=np.float64,
    )
    phase_rad = (
        2.0
        * np.pi
        * normalized_frequency
        * sample_indices
        + initial_phase_rad
    )
    return (
        np.sqrt(power_linear)
        * np.exp(1j * phase_rad)
    ).astype(np.complex128)
def generate_qpsk_signal(
    number_of_snapshots: int,
    *,
    power_linear: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Generate a random unit-energy QPSK sequence."""
    _validate_positive_integer(
        number_of_snapshots,
        name="number_of_snapshots",
    )
    _validate_nonnegative_finite(
        power_linear,
        name="power_linear",
    )
    if power_linear == 0.0:
        return np.zeros(
            number_of_snapshots,
            dtype=np.complex128,
        )
    random_generator = np.random.default_rng(
        seed
    )
    symbol_indices = random_generator.integers(
        low=0,
        high=4,
        size=number_of_snapshots,
    )
    constellation = np.array(
        [
            1.0 + 1.0j,
            -1.0 + 1.0j,
            -1.0 - 1.0j,
            1.0 - 1.0j,
        ],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    return (
        np.sqrt(power_linear)
        * constellation[symbol_indices]
    ).astype(np.complex128)
def generate_awgn(
    shape: tuple[int, ...],
    *,
    noise_power_linear: float,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    Generate circular complex additive white Gaussian noise.
    ``noise_power_linear`` is the expected noise power per sensor.
    """
    if not isinstance(shape, tuple) or len(shape) < 1:
        raise TypeError(
            "shape must be a non-empty tuple."
        )
    for dimension in shape:
        _validate_positive_integer(
            dimension,
            name="shape dimension",
        )
    _validate_nonnegative_finite(
        noise_power_linear,
        name="noise_power_linear",
    )
    if noise_power_linear == 0.0:
        return np.zeros(
            shape,
            dtype=np.complex128,
        )
    random_generator = np.random.default_rng(
        seed
    )
    standard_deviation = np.sqrt(
        noise_power_linear / 2.0
    )
    return (
        random_generator.normal(
            loc=0.0,
            scale=standard_deviation,
            size=shape,
        )
        + 1j
        * random_generator.normal(
            loc=0.0,
            scale=standard_deviation,
            size=shape,
        )
    ).astype(np.complex128)
def generate_source_signal(
    signal_type: str,
    number_of_snapshots: int,
    power_linear: float,
    *,
    random_generator: np.random.Generator,
    normalized_frequency: float | None = None,
) -> NDArray[np.complex128]:
    """Generate a source waveform using a supported signal model."""
    normalized_type = (
        str(signal_type)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if normalized_type in {
        "complex_gaussian",
        "gaussian",
        "noise_like",
    }:
        signal = (
            random_generator.normal(
                size=number_of_snapshots
            )
            + 1j
            * random_generator.normal(
                size=number_of_snapshots
            )
        ) / np.sqrt(2.0)
        signal *= np.sqrt(power_linear)
        return signal.astype(
            np.complex128
        )
    if normalized_type in {
        "qpsk",
        "quadrature_phase_shift_keying",
    }:
        symbols = random_generator.integers(
            low=0,
            high=4,
            size=number_of_snapshots,
        )
        constellation = np.array(
            [
                1.0 + 1.0j,
                -1.0 + 1.0j,
                -1.0 - 1.0j,
                1.0 - 1.0j,
            ],
            dtype=np.complex128,
        ) / np.sqrt(2.0)
        return (
            np.sqrt(power_linear)
            * constellation[symbols]
        ).astype(np.complex128)
    if normalized_type in {
        "tone",
        "complex_tone",
        "sinusoid",
    }:
        if normalized_frequency is None:
            normalized_frequency = float(
                random_generator.uniform(
                    -0.25,
                    0.25,
                )
            )
        if not -0.5 <= normalized_frequency <= 0.5:
            raise ValueError(
                "normalized_frequency must lie between -0.5 and 0.5."
            )
        initial_phase_rad = float(
            random_generator.uniform(
                -np.pi,
                np.pi,
            )
        )
        sample_indices = np.arange(
            number_of_snapshots,
            dtype=np.float64,
        )
        return (
            np.sqrt(power_linear)
            * np.exp(
                1j
                * (
                    2.0
                    * np.pi
                    * normalized_frequency
                    * sample_indices
                    + initial_phase_rad
                )
            )
        ).astype(np.complex128)
    raise ValueError(
        "Unsupported signal_type. Supported values are "
        "'complex_gaussian', 'qpsk', and 'tone'."
    )
def simulate_ula_snapshots(
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    number_of_snapshots: int,
    sources: list[SourceDefinition] | tuple[SourceDefinition, ...],
    *,
    noise_power_db: float = 0.0,
    seed: int | None = None,
    remove_sample_mean: bool = False,
) -> SnapshotSimulationResult:
    """
    Simulate narrowband far-field signals received by a ULA.
    The array model is
        X = A S + N,
    where:
    - ``A`` is the steering matrix,
    - ``S`` contains source waveforms,
    - ``N`` is circular complex AWGN.
    Source powers and noise power are absolute relative simulation
    levels in dB. For example, with ``noise_power_db=0``:
    - source power 10 dB corresponds to SNR = 10 dB,
    - source power 20 dB corresponds to SNR = 20 dB.
    """
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
    _validate_positive_integer(
        number_of_snapshots,
        name="number_of_snapshots",
    )
    if not isinstance(
        sources,
        (list, tuple),
    ):
        raise TypeError(
            "sources must be a list or tuple of SourceDefinition objects."
        )
    if len(sources) < 1:
        raise ValueError(
            "At least one source is required."
        )
    for source in sources:
        _validate_source_definition(
            source
        )
    if not np.isfinite(noise_power_db):
        raise ValueError(
            "noise_power_db must be finite."
        )
    random_generator = np.random.default_rng(
        seed
    )
    number_of_sources = len(sources)
    steering_matrix = np.column_stack(
        [
            steering_vector_ula(
                number_of_elements=number_of_elements,
                frequency_hz=frequency_hz,
                element_spacing_m=element_spacing_m,
                angle_deg=source.angle_deg,
            )
            for source in sources
        ]
    ).astype(np.complex128)
    source_powers_linear = db_to_linear_power(
        np.array(
            [
                source.power_db
                for source in sources
            ],
            dtype=np.float64,
        )
    )
    source_signals = np.zeros(
        (
            number_of_sources,
            number_of_snapshots,
        ),
        dtype=np.complex128,
    )
    correlation_group_signals: dict[
        str,
        NDArray[np.complex128],
    ] = {}
    for source_index, source in enumerate(
        sources
    ):
        source_signal = generate_source_signal(
            signal_type=source.signal_type,
            number_of_snapshots=number_of_snapshots,
            power_linear=float(
                source_powers_linear[
                    source_index
                ]
            ),
            random_generator=random_generator,
        )
        if source.correlation_group is not None:
            correlation_group = (
                source.correlation_group
            )
            correlation_coefficient = complex(
                source.correlation_coefficient
            )
            coefficient_magnitude = abs(
                correlation_coefficient
            )
            if coefficient_magnitude > 1.0:
                raise ValueError(
                    "The magnitude of correlation_coefficient "
                    "cannot exceed 1."
                )
            if (
                correlation_group
                not in correlation_group_signals
            ):
                reference_signal = (
                    source_signal.copy()
                )
                reference_power = float(
                    source_powers_linear[
                        source_index
                    ]
                )
                if reference_power > 0.0:
                    reference_signal /= np.sqrt(
                        reference_power
                    )
                correlation_group_signals[
                    correlation_group
                ] = reference_signal
            else:
                reference_signal = (
                    correlation_group_signals[
                        correlation_group
                    ]
                )
                independent_signal = (
                    source_signal.copy()
                )
                source_power = float(
                    source_powers_linear[
                        source_index
                    ]
                )
                if source_power > 0.0:
                    independent_signal /= np.sqrt(
                        source_power
                    )
                residual_scale = np.sqrt(
                    max(
                        0.0,
                        1.0
                        - coefficient_magnitude**2,
                    )
                )
                normalized_correlated_signal = (
                    correlation_coefficient
                    * reference_signal
                    + residual_scale
                    * independent_signal
                )
                source_signal = (
                    np.sqrt(source_power)
                    * normalized_correlated_signal
                )
        source_signals[
            source_index,
            :,
        ] = source_signal
    clean_snapshots = (
        steering_matrix
        @ source_signals
    )
    noise_power_linear = float(
        db_to_linear_power(
            noise_power_db
        )
    )
    noise_snapshots = (
        random_generator.normal(
            size=clean_snapshots.shape
        )
        + 1j
        * random_generator.normal(
            size=clean_snapshots.shape
        )
    ) * np.sqrt(
        noise_power_linear / 2.0
    )
    snapshots = (
        clean_snapshots
        + noise_snapshots
    ).astype(np.complex128)
    if remove_sample_mean:
        snapshots -= np.mean(
            snapshots,
            axis=1,
            keepdims=True,
        )
    sample_covariance = (
        snapshots
        @ snapshots.conj().T
    ) / number_of_snapshots
    sample_covariance = (
        sample_covariance
        + sample_covariance.conj().T
    ) / 2.0
    source_covariance = (
        source_signals
        @ source_signals.conj().T
    ) / number_of_snapshots
    theoretical_covariance = (
        steering_matrix
        @ source_covariance
        @ steering_matrix.conj().T
        + noise_power_linear
        * np.eye(
            number_of_elements,
            dtype=np.complex128,
        )
    )
    theoretical_covariance = (
        theoretical_covariance
        + theoretical_covariance.conj().T
    ) / 2.0
    return SnapshotSimulationResult(
        snapshots=np.asarray(
            snapshots,
            dtype=np.complex128,
        ),
        clean_snapshots=np.asarray(
            clean_snapshots,
            dtype=np.complex128,
        ),
        noise_snapshots=np.asarray(
            noise_snapshots,
            dtype=np.complex128,
        ),
        source_signals=np.asarray(
            source_signals,
            dtype=np.complex128,
        ),
        steering_matrix=np.asarray(
            steering_matrix,
            dtype=np.complex128,
        ),
        source_angles_deg=np.asarray(
            [
                source.angle_deg
                for source in sources
            ],
            dtype=np.float64,
        ),
        source_powers_linear=np.asarray(
            source_powers_linear,
            dtype=np.float64,
        ),
        source_labels=tuple(
            source.label
            for source in sources
        ),
        noise_power_linear=noise_power_linear,
        sample_covariance=np.asarray(
            sample_covariance,
            dtype=np.complex128,
        ),
        theoretical_covariance=np.asarray(
            theoretical_covariance,
            dtype=np.complex128,
        ),
        number_of_elements=number_of_elements,
        number_of_snapshots=number_of_snapshots,
    )
def calculate_empirical_sensor_power(
    snapshots: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Calculate mean received power for each array element."""
    data = _prepare_snapshot_matrix(
        snapshots
    )
    return np.mean(
        np.abs(data) ** 2,
        axis=1,
    ).astype(np.float64)
def calculate_empirical_covariance(
    snapshots: NDArray[np.complex128],
    *,
    remove_mean: bool = False,
    unbiased: bool = False,
) -> NDArray[np.complex128]:
    """Calculate a Hermitian sample covariance matrix."""
    data = _prepare_snapshot_matrix(
        snapshots
    )
    processed = data.copy()
    if remove_mean:
        processed -= np.mean(
            processed,
            axis=1,
            keepdims=True,
        )
    number_of_snapshots = (
        processed.shape[1]
    )
    if unbiased:
        if number_of_snapshots < 2:
            raise ValueError(
                "At least two snapshots are needed for "
                "an unbiased covariance estimate."
            )
        denominator = (
            number_of_snapshots - 1
        )
    else:
        denominator = number_of_snapshots
    covariance = (
        processed
        @ processed.conj().T
    ) / denominator
    return np.asarray(
        (
            covariance
            + covariance.conj().T
        )
        / 2.0,
        dtype=np.complex128,
    )
def calculate_covariance_error(
    estimated_covariance: NDArray[np.complex128],
    reference_covariance: NDArray[np.complex128],
) -> dict[str, float]:
    """Compare an estimated covariance matrix with a reference matrix."""
    estimated = _prepare_covariance_matrix(
        estimated_covariance,
        name="estimated_covariance",
    )
    reference = _prepare_covariance_matrix(
        reference_covariance,
        name="reference_covariance",
    )
    if estimated.shape != reference.shape:
        raise ValueError(
            "Covariance matrices must have the same shape."
        )
    error = estimated - reference
    reference_norm = float(
        np.linalg.norm(
            reference,
            ord="fro",
        )
    )
    error_norm = float(
        np.linalg.norm(
            error,
            ord="fro",
        )
    )
    normalized_error = (
        error_norm / reference_norm
        if reference_norm > 0.0
        else float("nan")
    )
    return {
        "frobenius_error": error_norm,
        "normalized_frobenius_error": float(
            normalized_error
        ),
        "maximum_absolute_error": float(
            np.max(
                np.abs(error)
            )
        ),
        "estimated_condition_number": float(
            np.linalg.cond(
                estimated
            )
        ),
        "reference_condition_number": float(
            np.linalg.cond(
                reference
            )
        ),
    }
def _validate_source_definition(
    source: SourceDefinition,
) -> None:
    """Validate one source definition."""
    if not isinstance(
        source,
        SourceDefinition,
    ):
        raise TypeError(
            "Every source must be a SourceDefinition object."
        )
    if not np.isfinite(source.angle_deg):
        raise ValueError(
            "Source angle must be finite."
        )
    if not -90.0 <= source.angle_deg <= 90.0:
        raise ValueError(
            "Source angle must lie between -90 and 90 degrees."
        )
    if not np.isfinite(source.power_db):
        raise ValueError(
            "Source power_db must be finite."
        )
    if not str(source.label).strip():
        raise ValueError(
            "Source label cannot be empty."
        )
    coefficient = complex(
        source.correlation_coefficient
    )
    if not np.isfinite(
        coefficient.real
    ) or not np.isfinite(
        coefficient.imag
    ):
        raise ValueError(
            "correlation_coefficient must be finite."
        )
    if abs(coefficient) > 1.0:
        raise ValueError(
            "The magnitude of correlation_coefficient "
            "cannot exceed 1."
        )
def _prepare_snapshot_matrix(
    snapshots: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Validate a complex snapshot matrix."""
    data = np.asarray(
        snapshots,
        dtype=np.complex128,
    )
    if data.ndim != 2:
        raise ValueError(
            "snapshots must have shape (elements, snapshots)."
        )
    if data.shape[0] < 1:
        raise ValueError(
            "snapshots must contain at least one sensor."
        )
    if data.shape[1] < 1:
        raise ValueError(
            "snapshots must contain at least one time snapshot."
        )
    if not np.all(
        np.isfinite(data.real)
    ) or not np.all(
        np.isfinite(data.imag)
    ):
        raise ValueError(
            "snapshots contains non-finite values."
        )
    return data
def _prepare_covariance_matrix(
    matrix: NDArray[np.complex128],
    *,
    name: str,
) -> NDArray[np.complex128]:
    """Validate and Hermitian-symmetrize a covariance matrix."""
    covariance = np.asarray(
        matrix,
        dtype=np.complex128,
    )
    if covariance.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional."
        )
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"{name} must be square."
        )
    if not np.all(
        np.isfinite(covariance.real)
    ) or not np.all(
        np.isfinite(covariance.imag)
    ):
        raise ValueError(
            f"{name} contains non-finite values."
        )
    return np.asarray(
        (
            covariance
            + covariance.conj().T
        )
        / 2.0,
        dtype=np.complex128,
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
    if not np.isfinite(value):
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
    if not np.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )
    if value < 0.0:
        raise ValueError(
            f"{name} cannot be negative."
        )