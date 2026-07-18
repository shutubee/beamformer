
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationEstimate:
    """Estimated element responses from a known reference source."""

    measured_response: NDArray[np.complex128]
    ideal_response: NDArray[np.complex128]
    relative_response: NDArray[np.complex128]
    gain_error_linear: NDArray[np.float64]
    gain_error_db: NDArray[np.float64]
    phase_error_rad: NDArray[np.float64]
    phase_error_deg: NDArray[np.float64]
    reference_element_index: int
    valid_element_mask: NDArray[np.bool_]
    number_of_snapshots: int


@dataclass(frozen=True)
class CalibrationCorrection:
    """Complex correction coefficients for array calibration."""

    correction_weights: NDArray[np.complex128]
    normalized_correction_weights: NDArray[np.complex128]
    maximum_correction_gain_db: float
    minimum_correction_gain_db: float
    reference_element_index: int


@dataclass(frozen=True)
class CalibrationValidation:
    """Residual errors after applying calibration coefficients."""

    corrected_response: NDArray[np.complex128]
    residual_gain_error_db: NDArray[np.float64]
    residual_phase_error_deg: NDArray[np.float64]
    rms_gain_error_db: float
    maximum_absolute_gain_error_db: float
    rms_phase_error_deg: float
    maximum_absolute_phase_error_deg: float
    valid_element_mask: NDArray[np.bool_]


def estimate_reference_response(
    snapshots: NDArray[np.complex128],
    reference_signal: NDArray[np.complex128],
    ideal_steering_vector: NDArray[np.complex128],
    *,
    reference_element_index: int = 0,
    minimum_signal_power: float = 1e-12,
) -> CalibrationEstimate:
    """
    Estimate element gain and phase errors using a known reference signal.

    The receiver model is

        x_n[k] = g_n a_n s[k] + v_n[k],

    where:

    - ``g_n`` is the unknown complex element response,
    - ``a_n`` is the ideal steering-vector value,
    - ``s[k]`` is the known calibration waveform,
    - ``v_n[k]`` is noise.

    A least-squares complex response estimate is obtained using

        h_n =
            sum_k x_n[k] conj(s[k])
            / sum_k |s[k]|².

    The estimated hardware response is then

        g_n = h_n / a_n.

    Results are made relative to a selected reference element so that
    common gain and phase offsets are removed.

    Parameters
    ----------
    snapshots:
        Complex array data with shape ``(N, K)``.

    reference_signal:
        Known transmitted calibration waveform with shape ``(K,)``.

    ideal_steering_vector:
        Ideal array steering vector for the calibration-source angle,
        with shape ``(N,)``.

    reference_element_index:
        Element used as the relative gain and phase reference.

    minimum_signal_power:
        Lower bound for the total reference-signal power.
    """

    data = _prepare_snapshots(
        snapshots
    )

    signal = _prepare_vector(
        reference_signal,
        name="reference_signal",
        minimum_size=2,
    )

    ideal_response = _prepare_vector(
        ideal_steering_vector,
        name="ideal_steering_vector",
        minimum_size=2,
    )

    number_of_elements, number_of_snapshots = data.shape

    if signal.size != number_of_snapshots:
        raise ValueError(
            "reference_signal length must match the snapshot count."
        )

    if ideal_response.size != number_of_elements:
        raise ValueError(
            "ideal_steering_vector length must match the number "
            "of array elements."
        )

    _validate_reference_index(
        reference_element_index=reference_element_index,
        number_of_elements=number_of_elements,
    )

    _validate_positive_finite(
        minimum_signal_power,
        name="minimum_signal_power",
    )

    reference_power = float(
        np.vdot(
            signal,
            signal,
        ).real
    )

    if reference_power < minimum_signal_power:
        raise ValueError(
            "The reference signal has insufficient energy for "
            "calibration."
        )

    measured_response = (
        data
        @ signal.conj()
    ) / reference_power

    ideal_magnitude = np.abs(
        ideal_response
    )

    valid_element_mask = (
        ideal_magnitude
        > np.finfo(np.float64).eps
    )

    if not valid_element_mask[
        reference_element_index
    ]:
        raise ValueError(
            "The selected reference element has zero ideal response."
        )

    hardware_response = np.full(
        number_of_elements,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )

    hardware_response[
        valid_element_mask
    ] = (
        measured_response[
            valid_element_mask
        ]
        / ideal_response[
            valid_element_mask
        ]
    )

    reference_response = hardware_response[
        reference_element_index
    ]

    if not np.isfinite(
        reference_response.real
    ) or not np.isfinite(
        reference_response.imag
    ):
        raise ValueError(
            "The reference-element response is not finite."
        )

    if np.isclose(
        np.abs(reference_response),
        0.0,
    ):
        raise ValueError(
            "The reference-element response is too small for "
            "relative calibration."
        )

    relative_response = np.full(
        number_of_elements,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )

    relative_response[
        valid_element_mask
    ] = (
        hardware_response[
            valid_element_mask
        ]
        / reference_response
    )

    gain_error_linear = np.full(
        number_of_elements,
        np.nan,
        dtype=np.float64,
    )

    gain_error_linear[
        valid_element_mask
    ] = np.abs(
        relative_response[
            valid_element_mask
        ]
    )

    minimum_linear = np.finfo(
        np.float64
    ).tiny

    gain_error_db = np.full(
        number_of_elements,
        np.nan,
        dtype=np.float64,
    )

    gain_error_db[
        valid_element_mask
    ] = (
        20.0
        * np.log10(
            np.maximum(
                gain_error_linear[
                    valid_element_mask
                ],
                minimum_linear,
            )
        )
    )

    phase_error_rad = np.full(
        number_of_elements,
        np.nan,
        dtype=np.float64,
    )

    phase_error_rad[
        valid_element_mask
    ] = np.angle(
        relative_response[
            valid_element_mask
        ]
    )

    phase_error_deg = np.rad2deg(
        phase_error_rad
    )

    return CalibrationEstimate(
        measured_response=np.asarray(
            measured_response,
            dtype=np.complex128,
        ),
        ideal_response=ideal_response,
        relative_response=relative_response,
        gain_error_linear=gain_error_linear,
        gain_error_db=gain_error_db,
        phase_error_rad=phase_error_rad,
        phase_error_deg=phase_error_deg,
        reference_element_index=reference_element_index,
        valid_element_mask=valid_element_mask,
        number_of_snapshots=number_of_snapshots,
    )


def estimate_response_from_covariance(
    covariance_matrix: NDArray[np.complex128],
    ideal_steering_vector: NDArray[np.complex128],
    *,
    reference_element_index: int = 0,
    dominant_eigenvector: bool = True,
) -> CalibrationEstimate:
    """
    Estimate relative element errors from a calibration covariance matrix.

    This method is useful when the transmitted calibration waveform is
    unavailable but the received field is dominated by one known source.

    The dominant eigenvector of the covariance matrix approximates the
    measured array response.

    Parameters
    ----------
    covariance_matrix:
        Spatial covariance matrix with shape ``(N, N)``.

    ideal_steering_vector:
        Ideal response for the known calibration-source direction.

    reference_element_index:
        Element used as the relative response reference.

    dominant_eigenvector:
        When True, use the eigenvector associated with the largest
        eigenvalue. This parameter is retained for explicitness.
    """

    covariance = _prepare_covariance(
        covariance_matrix
    )

    ideal_response = _prepare_vector(
        ideal_steering_vector,
        name="ideal_steering_vector",
        minimum_size=2,
    )

    number_of_elements = covariance.shape[0]

    if ideal_response.size != number_of_elements:
        raise ValueError(
            "ideal_steering_vector length must match the covariance "
            "matrix dimension."
        )

    _validate_reference_index(
        reference_element_index=reference_element_index,
        number_of_elements=number_of_elements,
    )

    if not dominant_eigenvector:
        raise ValueError(
            "Only dominant-eigenvector covariance calibration is "
            "currently supported."
        )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    dominant_index = int(
        np.argmax(
            np.real(
                eigenvalues
            )
        )
    )

    measured_response = eigenvectors[
        :,
        dominant_index,
    ].astype(np.complex128)

    reference_value = measured_response[
        reference_element_index
    ]

    if np.isclose(
        np.abs(reference_value),
        0.0,
    ):
        raise ValueError(
            "The dominant eigenvector has a near-zero value at the "
            "selected reference element."
        )

    measured_response /= reference_value

    ideal_reference = ideal_response[
        reference_element_index
    ]

    if np.isclose(
        np.abs(ideal_reference),
        0.0,
    ):
        raise ValueError(
            "The selected reference element has zero ideal response."
        )

    normalized_ideal = (
        ideal_response
        / ideal_reference
    )

    relative_response = (
        measured_response
        / normalized_ideal
    )

    gain_error_linear = np.abs(
        relative_response
    )

    gain_error_db = (
        20.0
        * np.log10(
            np.maximum(
                gain_error_linear,
                np.finfo(np.float64).tiny,
            )
        )
    )

    phase_error_rad = np.angle(
        relative_response
    )

    phase_error_deg = np.rad2deg(
        phase_error_rad
    )

    valid_mask = np.ones(
        number_of_elements,
        dtype=bool,
    )

    return CalibrationEstimate(
        measured_response=measured_response,
        ideal_response=normalized_ideal,
        relative_response=relative_response,
        gain_error_linear=gain_error_linear.astype(
            np.float64
        ),
        gain_error_db=gain_error_db.astype(
            np.float64
        ),
        phase_error_rad=phase_error_rad.astype(
            np.float64
        ),
        phase_error_deg=phase_error_deg.astype(
            np.float64
        ),
        reference_element_index=reference_element_index,
        valid_element_mask=valid_mask,
        number_of_snapshots=0,
    )


def calculate_correction_weights(
    calibration_estimate: CalibrationEstimate,
    *,
    maximum_gain_correction_db: float | None = 12.0,
    normalize_mode: str = "reference",
) -> CalibrationCorrection:
    """
    Calculate inverse complex correction coefficients.

    For an estimated relative hardware response ``g_n``, the ideal
    inverse correction is

        c_n = 1 / g_n.

    Parameters
    ----------
    calibration_estimate:
        Estimated relative gain and phase errors.

    maximum_gain_correction_db:
        Optional limit on the magnitude of correction amplification.
        This prevents excessive amplification of weak or damaged
        channels. Set to None to disable clipping.

    normalize_mode:
        Correction normalization method:

        - ``reference``: reference-element correction equals one
        - ``peak``: largest correction magnitude equals one
        - ``power``: correction-vector power equals element count
        - ``none``: no normalization
    """

    if not isinstance(
        calibration_estimate,
        CalibrationEstimate,
    ):
        raise TypeError(
            "calibration_estimate must be a CalibrationEstimate."
        )

    relative_response = np.asarray(
        calibration_estimate.relative_response,
        dtype=np.complex128,
    )

    valid_mask = np.asarray(
        calibration_estimate.valid_element_mask,
        dtype=bool,
    )

    if relative_response.shape != valid_mask.shape:
        raise ValueError(
            "Calibration estimate response and valid mask shapes "
            "do not match."
        )

    if maximum_gain_correction_db is not None:
        _validate_nonnegative_finite(
            maximum_gain_correction_db,
            name="maximum_gain_correction_db",
        )

    correction_weights = np.zeros_like(
        relative_response,
        dtype=np.complex128,
    )

    safe_response_mask = (
        valid_mask
        & np.isfinite(
            relative_response.real
        )
        & np.isfinite(
            relative_response.imag
        )
        & (
            np.abs(
                relative_response
            )
            > np.finfo(np.float64).eps
        )
    )

    if not np.any(
        safe_response_mask
    ):
        raise ValueError(
            "No valid element responses are available for correction."
        )

    correction_weights[
        safe_response_mask
    ] = (
        1.0
        / relative_response[
            safe_response_mask
        ]
    )

    if maximum_gain_correction_db is not None:
        maximum_gain_linear = 10.0 ** (
            maximum_gain_correction_db
            / 20.0
        )

        correction_magnitude = np.abs(
            correction_weights
        )

        clipping_mask = (
            correction_magnitude
            > maximum_gain_linear
        )

        correction_weights[
            clipping_mask
        ] *= (
            maximum_gain_linear
            / correction_magnitude[
                clipping_mask
            ]
        )

    normalized_weights = _normalize_correction_weights(
        correction_weights=correction_weights,
        valid_mask=safe_response_mask,
        reference_element_index=(
            calibration_estimate.reference_element_index
        ),
        normalization=normalize_mode,
    )

    valid_magnitudes = np.abs(
        normalized_weights[
            safe_response_mask
        ]
    )

    valid_gain_db = (
        20.0
        * np.log10(
            np.maximum(
                valid_magnitudes,
                np.finfo(np.float64).tiny,
            )
        )
    )

    return CalibrationCorrection(
        correction_weights=correction_weights,
        normalized_correction_weights=normalized_weights,
        maximum_correction_gain_db=float(
            np.max(
                valid_gain_db
            )
        ),
        minimum_correction_gain_db=float(
            np.min(
                valid_gain_db
            )
        ),
        reference_element_index=(
            calibration_estimate.reference_element_index
        ),
    )


def apply_calibration_to_weights(
    beamforming_weights: NDArray[np.complex128],
    correction_weights: NDArray[np.complex128],
    *,
    normalize_output: str = "none",
) -> NDArray[np.complex128]:
    """
    Apply calibration correction to transmit or receive weights.

    The corrected beamforming vector is

        w_corrected = w_ideal * c,

    where ``c`` contains inverse hardware-response coefficients.
    """

    beam_weights = _prepare_vector(
        beamforming_weights,
        name="beamforming_weights",
        minimum_size=2,
    )

    corrections = _prepare_vector(
        correction_weights,
        name="correction_weights",
        minimum_size=2,
        allow_zero=True,
    )

    if beam_weights.shape != corrections.shape:
        raise ValueError(
            "beamforming_weights and correction_weights must have "
            "the same shape."
        )

    corrected_weights = (
        beam_weights
        * corrections
    )

    return _normalize_complex_weights(
        corrected_weights,
        normalization=normalize_output,
    )


def apply_calibration_to_snapshots(
    snapshots: NDArray[np.complex128],
    correction_weights: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """
    Correct received array snapshots element by element.

    Parameters
    ----------
    snapshots:
        Complex data matrix with shape ``(N, K)``.

    correction_weights:
        One complex correction coefficient per sensor.
    """

    data = _prepare_snapshots(
        snapshots
    )

    corrections = _prepare_vector(
        correction_weights,
        name="correction_weights",
        minimum_size=2,
        allow_zero=True,
    )

    if corrections.size != data.shape[0]:
        raise ValueError(
            "correction_weights length must match the sensor count."
        )

    return (
        corrections[:, np.newaxis]
        * data
    ).astype(np.complex128)


def validate_calibration(
    measured_hardware_response: NDArray[np.complex128],
    correction_weights: NDArray[np.complex128],
    *,
    reference_element_index: int = 0,
    valid_element_mask: NDArray[np.bool_] | None = None,
) -> CalibrationValidation:
    """
    Measure residual relative gain and phase error after correction.

    The corrected hardware response is

        g_corrected = g_measured * c.

    A perfect correction produces equal complex responses for all valid
    elements after normalization to the reference element.
    """

    measured_response = _prepare_vector(
        measured_hardware_response,
        name="measured_hardware_response",
        minimum_size=2,
        allow_zero=True,
    )

    corrections = _prepare_vector(
        correction_weights,
        name="correction_weights",
        minimum_size=2,
        allow_zero=True,
    )

    if measured_response.shape != corrections.shape:
        raise ValueError(
            "measured_hardware_response and correction_weights must "
            "have the same shape."
        )

    number_of_elements = measured_response.size

    _validate_reference_index(
        reference_element_index=reference_element_index,
        number_of_elements=number_of_elements,
    )

    if valid_element_mask is None:
        valid_mask = (
            np.abs(
                measured_response
            )
            > np.finfo(np.float64).eps
        )
    else:
        valid_mask = np.asarray(
            valid_element_mask,
            dtype=bool,
        )

        if valid_mask.shape != (
            number_of_elements,
        ):
            raise ValueError(
                "valid_element_mask has an invalid shape."
            )

    valid_mask &= (
        np.abs(
            corrections
        )
        > 0.0
    )

    if not valid_mask[
        reference_element_index
    ]:
        raise ValueError(
            "The selected reference element is invalid after "
            "calibration."
        )

    corrected_response = (
        measured_response
        * corrections
    )

    reference_response = corrected_response[
        reference_element_index
    ]

    if np.isclose(
        np.abs(reference_response),
        0.0,
    ):
        raise ValueError(
            "The corrected reference response is too small."
        )

    normalized_response = np.full(
        number_of_elements,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )

    normalized_response[
        valid_mask
    ] = (
        corrected_response[
            valid_mask
        ]
        / reference_response
    )

    residual_gain_error_db = np.full(
        number_of_elements,
        np.nan,
        dtype=np.float64,
    )

    residual_gain_error_db[
        valid_mask
    ] = (
        20.0
        * np.log10(
            np.maximum(
                np.abs(
                    normalized_response[
                        valid_mask
                    ]
                ),
                np.finfo(np.float64).tiny,
            )
        )
    )

    residual_phase_error_deg = np.full(
        number_of_elements,
        np.nan,
        dtype=np.float64,
    )

    residual_phase_error_deg[
        valid_mask
    ] = np.rad2deg(
        np.angle(
            normalized_response[
                valid_mask
            ]
        )
    )

    valid_gain_errors = residual_gain_error_db[
        valid_mask
    ]

    valid_phase_errors = residual_phase_error_deg[
        valid_mask
    ]

    return CalibrationValidation(
        corrected_response=normalized_response,
        residual_gain_error_db=residual_gain_error_db,
        residual_phase_error_deg=residual_phase_error_deg,
        rms_gain_error_db=float(
            np.sqrt(
                np.mean(
                    valid_gain_errors**2
                )
            )
        ),
        maximum_absolute_gain_error_db=float(
            np.max(
                np.abs(
                    valid_gain_errors
                )
            )
        ),
        rms_phase_error_deg=float(
            np.sqrt(
                np.mean(
                    valid_phase_errors**2
                )
            )
        ),
        maximum_absolute_phase_error_deg=float(
            np.max(
                np.abs(
                    valid_phase_errors
                )
            )
        ),
        valid_element_mask=valid_mask,
    )


def simulate_calibration_measurement(
    ideal_steering_vector: NDArray[np.complex128],
    reference_signal: NDArray[np.complex128],
    *,
    gain_error_std_db: float = 0.5,
    phase_error_std_deg: float = 5.0,
    noise_power_linear: float = 0.01,
    failed_element_indices: (
        list[int] | NDArray[np.int64] | None
    ) = None,
    seed: int | None = None,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """
    Simulate a calibration-source measurement.

    Returns
    -------
    tuple
        Snapshot matrix and the true complex hardware-response vector.
    """

    ideal_response = _prepare_vector(
        ideal_steering_vector,
        name="ideal_steering_vector",
        minimum_size=2,
    )

    reference = _prepare_vector(
        reference_signal,
        name="reference_signal",
        minimum_size=2,
    )

    _validate_nonnegative_finite(
        gain_error_std_db,
        name="gain_error_std_db",
    )

    _validate_nonnegative_finite(
        phase_error_std_deg,
        name="phase_error_std_deg",
    )

    _validate_nonnegative_finite(
        noise_power_linear,
        name="noise_power_linear",
    )

    number_of_elements = ideal_response.size
    number_of_snapshots = reference.size

    random_generator = np.random.default_rng(
        seed
    )

    gain_error_db = random_generator.normal(
        loc=0.0,
        scale=gain_error_std_db,
        size=number_of_elements,
    )

    phase_error_deg = random_generator.normal(
        loc=0.0,
        scale=phase_error_std_deg,
        size=number_of_elements,
    )

    hardware_response = (
        10.0 ** (
            gain_error_db / 20.0
        )
        * np.exp(
            1j
            * np.deg2rad(
                phase_error_deg
            )
        )
    ).astype(np.complex128)

    if failed_element_indices is not None:
        failed_indices = np.asarray(
            failed_element_indices,
            dtype=np.int64,
        )

        if failed_indices.ndim != 1:
            raise ValueError(
                "failed_element_indices must be one-dimensional."
            )

        if np.any(
            failed_indices < 0
        ) or np.any(
            failed_indices >= number_of_elements
        ):
            raise IndexError(
                "failed_element_indices contains an invalid index."
            )

        hardware_response[
            failed_indices
        ] = 0.0 + 0.0j

    clean_snapshots = (
        hardware_response[:, np.newaxis]
        * ideal_response[:, np.newaxis]
        * reference[np.newaxis, :]
    )

    noise_standard_deviation = np.sqrt(
        noise_power_linear / 2.0
    )

    noise = (
        random_generator.normal(
            loc=0.0,
            scale=noise_standard_deviation,
            size=(
                number_of_elements,
                number_of_snapshots,
            ),
        )
        + 1j
        * random_generator.normal(
            loc=0.0,
            scale=noise_standard_deviation,
            size=(
                number_of_elements,
                number_of_snapshots,
            ),
        )
    )

    snapshots = (
        clean_snapshots
        + noise
    ).astype(np.complex128)

    return snapshots, hardware_response


def _normalize_correction_weights(
    correction_weights: NDArray[np.complex128],
    valid_mask: NDArray[np.bool_],
    reference_element_index: int,
    normalization: str,
) -> NDArray[np.complex128]:
    """Normalize correction coefficients using the selected method."""

    normalized = np.asarray(
        correction_weights,
        dtype=np.complex128,
    ).copy()

    normalized_mode = str(
        normalization
    ).strip().lower()

    valid_weights = normalized[
        valid_mask
    ]

    if normalized_mode == "reference":
        reference_value = normalized[
            reference_element_index
        ]

        if np.isclose(
            np.abs(reference_value),
            0.0,
        ):
            raise ValueError(
                "Reference correction coefficient is zero."
            )

        normalized[
            valid_mask
        ] /= reference_value

    elif normalized_mode == "peak":
        peak_magnitude = float(
            np.max(
                np.abs(
                    valid_weights
                )
            )
        )

        if np.isclose(
            peak_magnitude,
            0.0,
        ):
            raise ValueError(
                "Correction weights have zero magnitude."
            )

        normalized[
            valid_mask
        ] /= peak_magnitude

    elif normalized_mode == "power":
        current_power = float(
            np.sum(
                np.abs(
                    valid_weights
                ) ** 2
            )
        )

        target_power = float(
            np.count_nonzero(
                valid_mask
            )
        )

        if np.isclose(
            current_power,
            0.0,
        ):
            raise ValueError(
                "Correction weights have zero power."
            )

        normalized[
            valid_mask
        ] *= np.sqrt(
            target_power
            / current_power
        )

    elif normalized_mode == "none":
        pass

    else:
        raise ValueError(
            "normalization must be 'reference', 'peak', 'power', "
            "or 'none'."
        )

    return normalized


def _normalize_complex_weights(
    weights: NDArray[np.complex128],
    *,
    normalization: str,
) -> NDArray[np.complex128]:
    """Normalize a complex beamforming-weight vector."""

    prepared = np.asarray(
        weights,
        dtype=np.complex128,
    ).copy()

    normalized_mode = str(
        normalization
    ).strip().lower()

    nonzero_mask = np.abs(
        prepared
    ) > 0.0

    if not np.any(
        nonzero_mask
    ):
        raise ValueError(
            "The corrected weight vector contains only zeros."
        )

    if normalized_mode == "none":
        return prepared

    if normalized_mode == "peak":
        prepared /= np.max(
            np.abs(
                prepared
            )
        )

    elif normalized_mode == "power":
        prepared *= np.sqrt(
            np.count_nonzero(
                nonzero_mask
            )
            / np.sum(
                np.abs(
                    prepared
                ) ** 2
            )
        )

    elif normalized_mode == "sum":
        magnitude_sum = float(
            np.sum(
                np.abs(
                    prepared
                )
            )
        )

        if np.isclose(
            magnitude_sum,
            0.0,
        ):
            raise ValueError(
                "Cannot sum-normalize a zero-magnitude vector."
            )

        prepared /= magnitude_sum

    else:
        raise ValueError(
            "normalize_output must be 'none', 'peak', 'power', "
            "or 'sum'."
        )

    return prepared


def _prepare_snapshots(
    snapshots: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Validate an array snapshot matrix."""

    data = np.asarray(
        snapshots,
        dtype=np.complex128,
    )

    if data.ndim != 2:
        raise ValueError(
            "snapshots must have shape (elements, snapshots)."
        )

    if data.shape[0] < 2:
        raise ValueError(
            "snapshots must contain at least two array elements."
        )

    if data.shape[1] < 2:
        raise ValueError(
            "snapshots must contain at least two time snapshots."
        )

    if not np.all(
        np.isfinite(
            data.real
        )
    ) or not np.all(
        np.isfinite(
            data.imag
        )
    ):
        raise ValueError(
            "snapshots contains non-finite values."
        )

    return data


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

    return (
        covariance
        + covariance.conj().T
    ) / 2.0


def _prepare_vector(
    vector: NDArray[np.complex128],
    *,
    name: str,
    minimum_size: int,
    allow_zero: bool = False,
) -> NDArray[np.complex128]:
    """Validate a one-dimensional complex vector."""

    prepared = np.asarray(
        vector,
        dtype=np.complex128,
    )

    if prepared.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if prepared.size < minimum_size:
        raise ValueError(
            f"{name} must contain at least {minimum_size} values."
        )

    if not np.all(
        np.isfinite(
            prepared.real
        )
    ) or not np.all(
        np.isfinite(
            prepared.imag
        )
    ):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if not allow_zero and np.allclose(
        np.abs(
            prepared
        ),
        0.0,
    ):
        raise ValueError(
            f"{name} cannot contain only zeros."
        )

    return prepared.copy()


def _validate_reference_index(
    reference_element_index: int,
    number_of_elements: int,
) -> None:
    """Validate the selected calibration reference element."""

    if isinstance(
        reference_element_index,
        bool,
    ) or not isinstance(
        reference_element_index,
        (int, np.integer),
    ):
        raise TypeError(
            "reference_element_index must be an integer."
        )

    if not 0 <= reference_element_index < number_of_elements:
        raise IndexError(
            "reference_element_index is outside the element range."
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

