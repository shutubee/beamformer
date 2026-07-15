"""Hardware-imperfection models for phased antenna arrays."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
@dataclass(frozen=True)
class ImperfectionResult:
    """Element-level weights after hardware imperfections are applied."""
    ideal_weights: NDArray[np.complex128]
    imperfect_weights: NDArray[np.complex128]
    gain_error_db: NDArray[np.float64]
    phase_error_deg: NDArray[np.float64]
    active_mask: NDArray[np.bool_]
    failed_element_indices: NDArray[np.int64]
    quantization_bits: int | None
    @property
    def number_of_elements(self) -> int:
        """Return the number of physical array elements."""
        return int(self.ideal_weights.size)
    @property
    def number_of_active_elements(self) -> int:
        """Return the number of active elements."""
        return int(np.count_nonzero(self.active_mask))
    @property
    def failure_fraction(self) -> float:
        """Return the fraction of failed array elements."""
        return float(
            self.failed_element_indices.size
            / self.number_of_elements
        )
@dataclass(frozen=True)
class MonteCarloImperfectionSet:
    """Collection of imperfect weight realizations."""
    weight_realizations: NDArray[np.complex128]
    gain_error_db: NDArray[np.float64]
    phase_error_deg: NDArray[np.float64]
    active_masks: NDArray[np.bool_]
    quantization_bits: int | None
    seed: int | None
    @property
    def number_of_trials(self) -> int:
        """Return the number of Monte Carlo trials."""
        return int(self.weight_realizations.shape[0])
    @property
    def number_of_elements(self) -> int:
        """Return the number of elements in each trial."""
        return int(self.weight_realizations.shape[1])
def apply_array_imperfections(
    ideal_weights: NDArray[np.complex128],
    *,
    gain_error_std_db: float = 0.0,
    phase_error_std_deg: float = 0.0,
    failed_element_indices: (
        list[int] | NDArray[np.int64] | None
    ) = None,
    random_failure_probability: float = 0.0,
    quantization_bits: int | None = None,
    seed: int | None = None,
    preserve_total_power: bool = False,
) -> ImperfectionResult:
    """
    Apply practical hardware imperfections to complex array weights.
    Parameters
    ----------
    ideal_weights:
        Ideal complex excitation weights.
    gain_error_std_db:
        Standard deviation of independent Gaussian gain error in dB.
    phase_error_std_deg:
        Standard deviation of independent Gaussian phase error in
        degrees.
    failed_element_indices:
        Optional explicit indices of failed elements.
    random_failure_probability:
        Independent probability that each element fails.
    quantization_bits:
        Optional phase-shifter resolution. A B-bit phase shifter has
        ``2**B`` states over 360 degrees.
    seed:
        Random seed for repeatable simulations.
    preserve_total_power:
        When True, rescale active imperfect weights so their total power
        equals the total power of the ideal weights.
    Returns
    -------
    ImperfectionResult
        Imperfect weights and the generated error vectors.
    """
    prepared_weights = _prepare_weights(
        ideal_weights
    )
    number_of_elements = prepared_weights.size
    _validate_nonnegative_finite(
        gain_error_std_db,
        name="gain_error_std_db",
    )
    _validate_nonnegative_finite(
        phase_error_std_deg,
        name="phase_error_std_deg",
    )
    _validate_probability(
        random_failure_probability,
        name="random_failure_probability",
    )
    _validate_quantization_bits(
        quantization_bits
    )
    random_generator = np.random.default_rng(
        seed
    )
    gain_error_db = random_generator.normal(
        loc=0.0,
        scale=gain_error_std_db,
        size=number_of_elements,
    ).astype(np.float64)
    phase_error_deg = random_generator.normal(
        loc=0.0,
        scale=phase_error_std_deg,
        size=number_of_elements,
    ).astype(np.float64)
    active_mask = np.ones(
        number_of_elements,
        dtype=bool,
    )
    explicit_failures = _prepare_failed_indices(
        failed_element_indices=failed_element_indices,
        number_of_elements=number_of_elements,
    )
    active_mask[explicit_failures] = False
    if random_failure_probability > 0.0:
        random_failures = (
            random_generator.random(
                number_of_elements
            )
            < random_failure_probability
        )
        active_mask[random_failures] = False
    if not np.any(active_mask):
        raise ValueError(
            "All antenna elements failed. At least one active "
            "element is required."
        )
    ideal_amplitude = np.abs(
        prepared_weights
    )
    ideal_phase_rad = np.angle(
        prepared_weights
    )
    gain_scale = 10.0 ** (
        gain_error_db / 20.0
    )
    imperfect_amplitude = (
        ideal_amplitude * gain_scale
    )
    imperfect_phase_rad = (
        ideal_phase_rad
        + np.deg2rad(phase_error_deg)
    )
    if quantization_bits is not None:
        imperfect_phase_rad = quantize_phase(
            phase_rad=imperfect_phase_rad,
            number_of_bits=quantization_bits,
        )
    imperfect_weights = (
        imperfect_amplitude
        * np.exp(1j * imperfect_phase_rad)
    ).astype(np.complex128)
    imperfect_weights[~active_mask] = (
        0.0 + 0.0j
    )
    if preserve_total_power:
        imperfect_weights = _match_total_power(
            reference_weights=prepared_weights,
            weights_to_scale=imperfect_weights,
        )
    failed_indices = np.flatnonzero(
        ~active_mask
    ).astype(np.int64)
    return ImperfectionResult(
        ideal_weights=prepared_weights,
        imperfect_weights=imperfect_weights,
        gain_error_db=gain_error_db,
        phase_error_deg=phase_error_deg,
        active_mask=active_mask,
        failed_element_indices=failed_indices,
        quantization_bits=quantization_bits,
    )
def generate_monte_carlo_imperfections(
    ideal_weights: NDArray[np.complex128],
    number_of_trials: int,
    *,
    gain_error_std_db: float = 0.0,
    phase_error_std_deg: float = 0.0,
    random_failure_probability: float = 0.0,
    quantization_bits: int | None = None,
    seed: int | None = None,
    preserve_total_power: bool = False,
) -> MonteCarloImperfectionSet:
    """
    Generate multiple independent hardware-imperfection realizations.
    Parameters
    ----------
    ideal_weights:
        Ideal element excitation vector.
    number_of_trials:
        Number of Monte Carlo realizations.
    gain_error_std_db:
        Standard deviation of element gain mismatch in dB.
    phase_error_std_deg:
        Standard deviation of phase error in degrees.
    random_failure_probability:
        Independent failure probability for each element.
    quantization_bits:
        Optional phase-shifter resolution.
    seed:
        Random seed for reproducibility.
    preserve_total_power:
        Match each realization's total power to the ideal total power.
    """
    prepared_weights = _prepare_weights(
        ideal_weights
    )
    _validate_positive_integer(
        number_of_trials,
        name="number_of_trials",
    )
    _validate_nonnegative_finite(
        gain_error_std_db,
        name="gain_error_std_db",
    )
    _validate_nonnegative_finite(
        phase_error_std_deg,
        name="phase_error_std_deg",
    )
    _validate_probability(
        random_failure_probability,
        name="random_failure_probability",
    )
    _validate_quantization_bits(
        quantization_bits
    )
    number_of_elements = prepared_weights.size
    random_generator = np.random.default_rng(
        seed
    )
    gain_error_db = random_generator.normal(
        loc=0.0,
        scale=gain_error_std_db,
        size=(
            number_of_trials,
            number_of_elements,
        ),
    ).astype(np.float64)
    phase_error_deg = random_generator.normal(
        loc=0.0,
        scale=phase_error_std_deg,
        size=(
            number_of_trials,
            number_of_elements,
        ),
    ).astype(np.float64)
    active_masks = np.ones(
        (
            number_of_trials,
            number_of_elements,
        ),
        dtype=bool,
    )
    if random_failure_probability > 0.0:
        active_masks = (
            random_generator.random(
                (
                    number_of_trials,
                    number_of_elements,
                )
            )
            >= random_failure_probability
        )
        all_failed_trials = np.flatnonzero(
            ~np.any(active_masks, axis=1)
        )
        for trial_index in all_failed_trials:
            restored_index = int(
                random_generator.integers(
                    0,
                    number_of_elements,
                )
            )
            active_masks[
                trial_index,
                restored_index,
            ] = True
    ideal_amplitude = np.abs(
        prepared_weights
    )[np.newaxis, :]
    ideal_phase_rad = np.angle(
        prepared_weights
    )[np.newaxis, :]
    gain_scale = 10.0 ** (
        gain_error_db / 20.0
    )
    amplitude_realizations = (
        ideal_amplitude * gain_scale
    )
    phase_realizations_rad = (
        ideal_phase_rad
        + np.deg2rad(phase_error_deg)
    )
    if quantization_bits is not None:
        phase_realizations_rad = quantize_phase(
            phase_rad=phase_realizations_rad,
            number_of_bits=quantization_bits,
        )
    weight_realizations = (
        amplitude_realizations
        * np.exp(1j * phase_realizations_rad)
    ).astype(np.complex128)
    weight_realizations[~active_masks] = (
        0.0 + 0.0j
    )
    if preserve_total_power:
        reference_power = float(
            np.sum(
                np.abs(prepared_weights) ** 2
            )
        )
        realization_power = np.sum(
            np.abs(weight_realizations) ** 2,
            axis=1,
        )
        scale = np.ones(
            number_of_trials,
            dtype=np.float64,
        )
        valid_power = realization_power > 0.0
        scale[valid_power] = np.sqrt(
            reference_power
            / realization_power[valid_power]
        )
        weight_realizations *= (
            scale[:, np.newaxis]
        )
    return MonteCarloImperfectionSet(
        weight_realizations=weight_realizations,
        gain_error_db=gain_error_db,
        phase_error_deg=phase_error_deg,
        active_masks=active_masks,
        quantization_bits=quantization_bits,
        seed=seed,
    )
def quantize_phase(
    phase_rad: NDArray[np.float64] | float,
    number_of_bits: int,
) -> NDArray[np.float64]:
    """
    Quantize phase values to a finite number of phase-shifter states.
    Returned phases are wrapped to the interval [-π, π).
    """
    _validate_quantization_bits(
        number_of_bits
    )
    phase = np.asarray(
        phase_rad,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(phase)):
        raise ValueError(
            "phase_rad contains non-finite values."
        )
    number_of_states = 2**number_of_bits
    phase_step_rad = (
        2.0 * np.pi / number_of_states
    )
    phase_zero_to_two_pi = np.mod(
        phase,
        2.0 * np.pi,
    )
    quantized_phase = (
        np.round(
            phase_zero_to_two_pi
            / phase_step_rad
        )
        * phase_step_rad
    )
    return wrap_phase_rad(
        quantized_phase
    )
def wrap_phase_rad(
    phase_rad: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Wrap phase values to the interval [-π, π)."""
    phase = np.asarray(
        phase_rad,
        dtype=np.float64,
    )
    return (
        phase + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi
def calculate_weight_error_metrics(
    ideal_weights: NDArray[np.complex128],
    imperfect_weights: NDArray[np.complex128],
) -> dict[str, float]:
    """
    Calculate summary error metrics between two weight vectors.
    Returns
    -------
    dict
        RMS amplitude error, RMS phase error, failed-element count,
        and normalized complex-weight error.
    """
    ideal = _prepare_weights(
        ideal_weights
    )
    imperfect = np.asarray(
        imperfect_weights,
        dtype=np.complex128,
    )
    if imperfect.shape != ideal.shape:
        raise ValueError(
            "ideal_weights and imperfect_weights must have "
            "the same shape."
        )
    if not np.all(
        np.isfinite(imperfect.real)
    ) or not np.all(
        np.isfinite(imperfect.imag)
    ):
        raise ValueError(
            "imperfect_weights contains non-finite values."
        )
    ideal_amplitude = np.abs(
        ideal
    )
    imperfect_amplitude = np.abs(
        imperfect
    )
    amplitude_error = (
        imperfect_amplitude
        - ideal_amplitude
    )
    active_comparison_mask = (
        imperfect_amplitude > 0.0
    ) & (
        ideal_amplitude > 0.0
    )
    phase_error_deg = np.zeros(
        ideal.size,
        dtype=np.float64,
    )
    if np.any(active_comparison_mask):
        phase_difference = np.angle(
            imperfect[active_comparison_mask]
            * np.conj(
                ideal[active_comparison_mask]
            )
        )
        phase_error_deg[
            active_comparison_mask
        ] = np.rad2deg(
            phase_difference
        )
        rms_phase_error_deg = float(
            np.sqrt(
                np.mean(
                    phase_error_deg[
                        active_comparison_mask
                    ] ** 2
                )
            )
        )
    else:
        rms_phase_error_deg = float(
            "nan"
        )
    reference_norm = float(
        np.linalg.norm(ideal)
    )
    complex_error_norm = float(
        np.linalg.norm(
            imperfect - ideal
        )
    )
    normalized_complex_error = (
        complex_error_norm
        / reference_norm
    )
    return {
        "rms_amplitude_error": float(
            np.sqrt(
                np.mean(
                    amplitude_error**2
                )
            )
        ),
        "maximum_amplitude_error": float(
            np.max(
                np.abs(amplitude_error)
            )
        ),
        "rms_phase_error_deg": (
            rms_phase_error_deg
        ),
        "failed_element_count": float(
            np.count_nonzero(
                imperfect_amplitude == 0.0
            )
        ),
        "normalized_complex_weight_error": float(
            normalized_complex_error
        ),
    }
def _match_total_power(
    reference_weights: NDArray[np.complex128],
    weights_to_scale: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Scale a weight vector to match reference total power."""
    reference_power = float(
        np.sum(
            np.abs(reference_weights) ** 2
        )
    )
    current_power = float(
        np.sum(
            np.abs(weights_to_scale) ** 2
        )
    )
    if np.isclose(current_power, 0.0):
        raise ValueError(
            "Cannot normalize a zero-power weight vector."
        )
    scale = np.sqrt(
        reference_power / current_power
    )
    return np.asarray(
        weights_to_scale * scale,
        dtype=np.complex128,
    )
def _prepare_weights(
    weights: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Validate and prepare a complex weight vector."""
    prepared = np.asarray(
        weights,
        dtype=np.complex128,
    )
    if prepared.ndim != 1:
        raise ValueError(
            "weights must be one-dimensional."
        )
    if prepared.size < 2:
        raise ValueError(
            "weights must contain at least two values."
        )
    if not np.all(
        np.isfinite(prepared.real)
    ) or not np.all(
        np.isfinite(prepared.imag)
    ):
        raise ValueError(
            "weights contains non-finite values."
        )
    if np.allclose(
        np.abs(prepared),
        0.0,
    ):
        raise ValueError(
            "weights cannot contain only zeros."
        )
    return prepared.copy()
def _prepare_failed_indices(
    failed_element_indices: (
        list[int] | NDArray[np.int64] | None
    ),
    number_of_elements: int,
) -> NDArray[np.int64]:
    """Validate explicit failed-element indices."""
    if failed_element_indices is None:
        return np.empty(
            0,
            dtype=np.int64,
        )
    indices = np.asarray(
        failed_element_indices,
        dtype=np.int64,
    )
    if indices.ndim != 1:
        raise ValueError(
            "failed_element_indices must be one-dimensional."
        )
    if indices.size == 0:
        return indices
    if np.any(indices < 0) or np.any(
        indices >= number_of_elements
    ):
        raise IndexError(
            "failed_element_indices contains an invalid index."
        )
    if np.unique(indices).size != indices.size:
        raise ValueError(
            "failed_element_indices contains duplicate indices."
        )
    return indices
def _validate_quantization_bits(
    number_of_bits: int | None,
) -> None:
    """Validate phase-shifter resolution."""
    if number_of_bits is None:
        return
    if isinstance(
        number_of_bits,
        bool,
    ) or not isinstance(
        number_of_bits,
        (int, np.integer),
    ):
        raise TypeError(
            "quantization_bits must be an integer."
        )
    if number_of_bits < 1:
        raise ValueError(
            "quantization_bits must be at least 1."
        )
    if number_of_bits > 24:
        raise ValueError(
            "quantization_bits must not exceed 24."
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
def _validate_probability(
    value: float,
    *,
    name: str,
) -> None:
    """Validate a probability."""
    if not np.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must lie between 0 and 1."
        )