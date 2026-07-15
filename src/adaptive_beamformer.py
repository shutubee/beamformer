"""Adaptive beamforming algorithms for antenna arrays."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
@dataclass(frozen=True)
class CovarianceEstimate:
    """Sample covariance matrix and estimation metadata."""
    covariance_matrix: NDArray[np.complex128]
    number_of_elements: int
    number_of_snapshots: int
    diagonal_loading: float
@dataclass(frozen=True)
class AdaptiveWeights:
    """Adaptive beamforming weight result."""
    weights: NDArray[np.complex128]
    method: str
    output_power: float | None
    distortionless_response: complex | None
    condition_number: float | None
def steering_vector_ula(
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    angle_deg: float,
) -> NDArray[np.complex128]:
    """
    Generate a ULA receive steering vector.
    The array is assumed to lie along the x-axis, with angle measured
    from broadside.
    The steering vector is
        a(theta)[n] = exp(j n k d sin(theta)).
    """
    _validate_ula_inputs(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        angle_deg=angle_deg,
    )
    speed_of_light_m_s = 299_792_458.0
    wavelength_m = speed_of_light_m_s / frequency_hz
    wave_number_rad_m = 2.0 * np.pi / wavelength_m
    angle_rad = np.deg2rad(angle_deg)
    element_indices = np.arange(
        number_of_elements,
        dtype=np.float64,
    )
    phase_rad = (
        wave_number_rad_m
        * element_spacing_m
        * element_indices
        * np.sin(angle_rad)
    )
    return np.exp(
        1j * phase_rad
    ).astype(np.complex128)
def estimate_sample_covariance(
    snapshots: NDArray[np.complex128],
    *,
    remove_mean: bool = True,
    diagonal_loading: float = 0.0,
    normalize_by_snapshots: bool = True,
) -> CovarianceEstimate:
    """
    Estimate the spatial covariance matrix from array snapshots.
    Parameters
    ----------
    snapshots:
        Complex data matrix with shape ``(N, K)``, where N is the number
        of antenna elements and K is the number of snapshots.
    remove_mean:
        Remove the temporal mean from each sensor channel.
    diagonal_loading:
        Nonnegative loading coefficient. The applied loading is
            diagonal_loading * trace(R) / N.
    normalize_by_snapshots:
        Divide by K when True. Otherwise divide by K - 1.
    """
    data = np.asarray(
        snapshots,
        dtype=np.complex128,
    )
    if data.ndim != 2:
        raise ValueError(
            "snapshots must have shape (elements, snapshots)."
        )
    number_of_elements, number_of_snapshots = data.shape
    if number_of_elements < 2:
        raise ValueError(
            "At least two antenna elements are required."
        )
    if number_of_snapshots < 2:
        raise ValueError(
            "At least two snapshots are required."
        )
    if not np.all(
        np.isfinite(data.real)
    ) or not np.all(
        np.isfinite(data.imag)
    ):
        raise ValueError(
            "snapshots contains non-finite values."
        )
    if diagonal_loading < 0.0 or not np.isfinite(
        diagonal_loading
    ):
        raise ValueError(
            "diagonal_loading must be finite and nonnegative."
        )
    processed_data = data.copy()
    if remove_mean:
        processed_data -= np.mean(
            processed_data,
            axis=1,
            keepdims=True,
        )
    denominator = (
        number_of_snapshots
        if normalize_by_snapshots
        else number_of_snapshots - 1
    )
    covariance_matrix = (
        processed_data
        @ processed_data.conj().T
    ) / denominator
    covariance_matrix = (
        covariance_matrix
        + covariance_matrix.conj().T
    ) / 2.0
    loading_value = 0.0
    if diagonal_loading > 0.0:
        average_diagonal_power = float(
            np.real(
                np.trace(covariance_matrix)
            )
            / number_of_elements
        )
        loading_value = (
            diagonal_loading
            * average_diagonal_power
        )
        covariance_matrix = (
            covariance_matrix
            + loading_value
            * np.eye(
                number_of_elements,
                dtype=np.complex128,
            )
        )
    return CovarianceEstimate(
        covariance_matrix=np.asarray(
            covariance_matrix,
            dtype=np.complex128,
        ),
        number_of_elements=number_of_elements,
        number_of_snapshots=number_of_snapshots,
        diagonal_loading=float(
            loading_value
        ),
    )
def conventional_weights(
    steering_vector: NDArray[np.complex128],
    *,
    normalization: str = "distortionless",
) -> AdaptiveWeights:
    """
    Generate conventional delay-and-sum beamforming weights.
    Parameters
    ----------
    steering_vector:
        Desired-direction steering vector.
    normalization:
        Supported modes:
        - ``distortionless``: enforce wᴴa = 1
        - ``unit_norm``: enforce ||w||₂ = 1
        - ``none``: return the raw steering vector
    """
    vector = _prepare_vector(
        steering_vector,
        name="steering_vector",
    )
    normalized_mode = normalization.strip().lower()
    if normalized_mode == "distortionless":
        denominator = np.vdot(
            vector,
            vector,
        )
        if np.isclose(
            np.abs(denominator),
            0.0,
        ):
            raise ValueError(
                "The steering vector has zero energy."
            )
        weights = vector / denominator
    elif normalized_mode == "unit_norm":
        norm = np.linalg.norm(vector)
        if np.isclose(norm, 0.0):
            raise ValueError(
                "The steering vector has zero norm."
            )
        weights = vector / norm
    elif normalized_mode == "none":
        weights = vector.copy()
    else:
        raise ValueError(
            "normalization must be 'distortionless', "
            "'unit_norm', or 'none'."
        )
    distortionless_response = np.vdot(
        weights,
        vector,
    )
    return AdaptiveWeights(
        weights=np.asarray(
            weights,
            dtype=np.complex128,
        ),
        method="conventional",
        output_power=None,
        distortionless_response=complex(
            distortionless_response
        ),
        condition_number=None,
    )
def mvdr_weights(
    covariance_matrix: NDArray[np.complex128],
    steering_vector: NDArray[np.complex128],
    *,
    diagonal_loading: float = 0.0,
    use_pseudoinverse: bool = False,
) -> AdaptiveWeights:
    """
    Calculate MVDR/Capon beamforming weights.
    The MVDR solution is
        w = R⁻¹a / (aᴴR⁻¹a).
    It minimizes output power while preserving unity response in the
    desired steering direction.
    """
    covariance = _prepare_covariance(
        covariance_matrix
    )
    steering = _prepare_vector(
        steering_vector,
        name="steering_vector",
    )
    if covariance.shape[0] != steering.size:
        raise ValueError(
            "covariance_matrix and steering_vector dimensions "
            "do not match."
        )
    if diagonal_loading < 0.0 or not np.isfinite(
        diagonal_loading
    ):
        raise ValueError(
            "diagonal_loading must be finite and nonnegative."
        )
    loaded_covariance = covariance.copy()
    if diagonal_loading > 0.0:
        average_power = float(
            np.real(
                np.trace(covariance)
            )
            / covariance.shape[0]
        )
        loaded_covariance += (
            diagonal_loading
            * average_power
            * np.eye(
                covariance.shape[0],
                dtype=np.complex128,
            )
        )
    condition_number = float(
        np.linalg.cond(
            loaded_covariance
        )
    )
    if use_pseudoinverse:
        inverse_times_steering = (
            np.linalg.pinv(
                loaded_covariance
            )
            @ steering
        )
    else:
        try:
            inverse_times_steering = (
                np.linalg.solve(
                    loaded_covariance,
                    steering,
                )
            )
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "The covariance matrix is singular or ill-conditioned. "
                "Use diagonal loading or use_pseudoinverse=True."
            ) from error
    denominator = np.vdot(
        steering,
        inverse_times_steering,
    )
    if np.isclose(
        np.abs(denominator),
        0.0,
    ):
        raise ValueError(
            "MVDR normalization denominator is zero."
        )
    weights = (
        inverse_times_steering
        / denominator
    )
    output_power = float(
        np.real(
            np.vdot(
                weights,
                loaded_covariance @ weights,
            )
        )
    )
    distortionless_response = np.vdot(
        weights,
        steering,
    )
    return AdaptiveWeights(
        weights=np.asarray(
            weights,
            dtype=np.complex128,
        ),
        method="mvdr",
        output_power=output_power,
        distortionless_response=complex(
            distortionless_response
        ),
        condition_number=condition_number,
    )
def lcmv_weights(
    covariance_matrix: NDArray[np.complex128],
    constraint_matrix: NDArray[np.complex128],
    response_vector: NDArray[np.complex128],
    *,
    diagonal_loading: float = 0.0,
    use_pseudoinverse: bool = False,
) -> AdaptiveWeights:
    """
    Calculate linearly constrained minimum-variance weights.
    The LCMV solution is
        w = R⁻¹C(CᴴR⁻¹C)⁻¹f,
    where columns of C are constraint steering vectors and f contains
    their requested complex responses.
    Typical use:
    - desired direction response = 1
    - interference-direction responses = 0
    """
    covariance = _prepare_covariance(
        covariance_matrix
    )
    constraints = np.asarray(
        constraint_matrix,
        dtype=np.complex128,
    )
    responses = np.asarray(
        response_vector,
        dtype=np.complex128,
    )
    if constraints.ndim != 2:
        raise ValueError(
            "constraint_matrix must be two-dimensional."
        )
    if responses.ndim != 1:
        raise ValueError(
            "response_vector must be one-dimensional."
        )
    number_of_elements, number_of_constraints = (
        constraints.shape
    )
    if covariance.shape != (
        number_of_elements,
        number_of_elements,
    ):
        raise ValueError(
            "constraint_matrix row count must match the "
            "covariance-matrix dimension."
        )
    if responses.shape != (
        number_of_constraints,
    ):
        raise ValueError(
            "response_vector must contain one value per constraint."
        )
    if number_of_constraints < 1:
        raise ValueError(
            "At least one constraint is required."
        )
    if number_of_constraints > number_of_elements:
        raise ValueError(
            "The number of constraints cannot exceed the "
            "number of array elements."
        )
    if not np.all(
        np.isfinite(constraints.real)
    ) or not np.all(
        np.isfinite(constraints.imag)
    ):
        raise ValueError(
            "constraint_matrix contains non-finite values."
        )
    if not np.all(
        np.isfinite(responses.real)
    ) or not np.all(
        np.isfinite(responses.imag)
    ):
        raise ValueError(
            "response_vector contains non-finite values."
        )
    loaded_covariance = covariance.copy()
    if diagonal_loading > 0.0:
        average_power = float(
            np.real(
                np.trace(covariance)
            )
            / number_of_elements
        )
        loaded_covariance += (
            diagonal_loading
            * average_power
            * np.eye(
                number_of_elements,
                dtype=np.complex128,
            )
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
                "The covariance matrix is singular. Use diagonal "
                "loading or use_pseudoinverse=True."
            ) from error
    constraint_gram = (
        constraints.conj().T
        @ inverse_covariance
        @ constraints
    )
    if use_pseudoinverse:
        constraint_solution = (
            np.linalg.pinv(
                constraint_gram
            )
            @ responses
        )
    else:
        try:
            constraint_solution = (
                np.linalg.solve(
                    constraint_gram,
                    responses,
                )
            )
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "The constraint system is singular or contains "
                "dependent steering vectors."
            ) from error
    weights = (
        inverse_covariance
        @ constraints
        @ constraint_solution
    )
    output_power = float(
        np.real(
            np.vdot(
                weights,
                loaded_covariance @ weights,
            )
        )
    )
    achieved_responses = (
        constraints.conj().T
        @ weights
    )
    return AdaptiveWeights(
        weights=np.asarray(
            weights,
            dtype=np.complex128,
        ),
        method="lcmv",
        output_power=output_power,
        distortionless_response=complex(
            achieved_responses[0]
        ),
        condition_number=float(
            np.linalg.cond(
                loaded_covariance
            )
        ),
    )
def null_steering_weights(
    desired_steering_vector: NDArray[np.complex128],
    interference_steering_vectors: NDArray[np.complex128],
    *,
    covariance_matrix: NDArray[np.complex128] | None = None,
    diagonal_loading: float = 0.0,
    use_pseudoinverse: bool = True,
) -> AdaptiveWeights:
    """
    Generate weights with unity desired response and spatial nulls.
    Parameters
    ----------
    desired_steering_vector:
        Steering vector for the desired signal.
    interference_steering_vectors:
        Matrix with shape ``(N, M)`` containing one interference
        steering vector per column.
    covariance_matrix:
        Optional covariance matrix. Identity covariance is used when
        omitted, producing deterministic constrained null steering.
    """
    desired = _prepare_vector(
        desired_steering_vector,
        name="desired_steering_vector",
    )
    interference = np.asarray(
        interference_steering_vectors,
        dtype=np.complex128,
    )
    if interference.ndim == 1:
        interference = interference[
            :, np.newaxis
        ]
    if interference.ndim != 2:
        raise ValueError(
            "interference_steering_vectors must be one- or "
            "two-dimensional."
        )
    if interference.shape[0] != desired.size:
        raise ValueError(
            "Interference steering vectors must have the same "
            "length as the desired steering vector."
        )
    if not np.all(
        np.isfinite(interference.real)
    ) or not np.all(
        np.isfinite(interference.imag)
    ):
        raise ValueError(
            "interference_steering_vectors contains non-finite values."
        )
    constraints = np.column_stack(
        (
            desired,
            interference,
        )
    )
    responses = np.zeros(
        constraints.shape[1],
        dtype=np.complex128,
    )
    responses[0] = 1.0 + 0.0j
    if covariance_matrix is None:
        covariance = np.eye(
            desired.size,
            dtype=np.complex128,
        )
    else:
        covariance = covariance_matrix
    result = lcmv_weights(
        covariance_matrix=covariance,
        constraint_matrix=constraints,
        response_vector=responses,
        diagonal_loading=diagonal_loading,
        use_pseudoinverse=use_pseudoinverse,
    )
    return AdaptiveWeights(
        weights=result.weights,
        method="null_steering",
        output_power=result.output_power,
        distortionless_response=(
            result.distortionless_response
        ),
        condition_number=result.condition_number,
    )
def calculate_beamformer_response(
    weights: NDArray[np.complex128],
    steering_vectors: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """
    Calculate beamformer response for one or more steering vectors.
    Parameters
    ----------
    weights:
        Beamforming vector with shape ``(N,)``.
    steering_vectors:
        Steering-vector matrix with shape ``(N, M)`` or one vector with
        shape ``(N,)``.
    Returns
    -------
    numpy.ndarray
        Complex response ``wᴴA``.
    """
    prepared_weights = _prepare_vector(
        weights,
        name="weights",
    )
    vectors = np.asarray(
        steering_vectors,
        dtype=np.complex128,
    )
    if vectors.ndim == 1:
        vectors = vectors[:, np.newaxis]
    if vectors.ndim != 2:
        raise ValueError(
            "steering_vectors must be one- or two-dimensional."
        )
    if vectors.shape[0] != prepared_weights.size:
        raise ValueError(
            "steering_vectors row count must match weights length."
        )
    if not np.all(
        np.isfinite(vectors.real)
    ) or not np.all(
        np.isfinite(vectors.imag)
    ):
        raise ValueError(
            "steering_vectors contains non-finite values."
        )
    return (
        prepared_weights.conj()
        @ vectors
    ).astype(np.complex128)
def apply_beamformer(
    weights: NDArray[np.complex128],
    snapshots: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """
    Apply beamforming weights to received array snapshots.
    The output signal is
        y[k] = wᴴx[k].
    """
    prepared_weights = _prepare_vector(
        weights,
        name="weights",
    )
    data = np.asarray(
        snapshots,
        dtype=np.complex128,
    )
    if data.ndim != 2:
        raise ValueError(
            "snapshots must have shape (elements, snapshots)."
        )
    if data.shape[0] != prepared_weights.size:
        raise ValueError(
            "snapshots row count must match weights length."
        )
    if not np.all(
        np.isfinite(data.real)
    ) or not np.all(
        np.isfinite(data.imag)
    ):
        raise ValueError(
            "snapshots contains non-finite values."
        )
    return (
        prepared_weights.conj()
        @ data
    ).astype(np.complex128)
def _prepare_covariance(
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
            "covariance_matrix must describe at least two elements."
        )
    if not np.all(
        np.isfinite(covariance.real)
    ) or not np.all(
        np.isfinite(covariance.imag)
    ):
        raise ValueError(
            "covariance_matrix contains non-finite values."
        )
    return (
        covariance + covariance.conj().T
    ) / 2.0
def _prepare_vector(
    vector: NDArray[np.complex128],
    *,
    name: str,
) -> NDArray[np.complex128]:
    """Validate a finite, nonzero complex vector."""
    prepared = np.asarray(
        vector,
        dtype=np.complex128,
    )
    if prepared.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )
    if prepared.size < 2:
        raise ValueError(
            f"{name} must contain at least two values."
        )
    if not np.all(
        np.isfinite(prepared.real)
    ) or not np.all(
        np.isfinite(prepared.imag)
    ):
        raise ValueError(
            f"{name} contains non-finite values."
        )
    if np.allclose(
        np.abs(prepared),
        0.0,
    ):
        raise ValueError(
            f"{name} cannot contain only zeros."
        )
    return prepared.copy()
def _validate_ula_inputs(
    number_of_elements: int,
    frequency_hz: float,
    element_spacing_m: float,
    angle_deg: float,
) -> None:
    """Validate ULA steering-vector parameters."""
    if isinstance(
        number_of_elements,
        bool,
    ) or not isinstance(
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
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError(
            "frequency_hz must be finite and positive."
        )
    if not np.isfinite(
        element_spacing_m
    ) or element_spacing_m <= 0.0:
        raise ValueError(
            "element_spacing_m must be finite and positive."
        )
    if not np.isfinite(angle_deg):
        raise ValueError(
            "angle_deg must be finite."
        )
    if not -90.0 <= angle_deg <= 90.0:
        raise ValueError(
            "angle_deg must lie between -90 and 90 degrees."
        )