"""Tests for the uniform linear array beamformer."""
from __future__ import annotations
import numpy as np
import pytest
from src.beamformer import (
    SPEED_OF_LIGHT_M_S,
    calculate_ula_array_factor,
)
from src.steering import generate_steering_weights
FREQUENCY_HZ = 28e9
WAVELENGTH_M = SPEED_OF_LIGHT_M_S / FREQUENCY_HZ
ELEMENT_SPACING_M = WAVELENGTH_M / 2.0
def test_broadside_uniform_array_peaks_at_zero_degrees() -> None:
    """A uniformly excited broadside ULA should peak at 0°."""
    result = calculate_ula_array_factor(
        number_of_elements=16,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        steering_angle_deg=0.0,
    )
    peak_index = int(
        np.argmax(result.magnitude)
    )
    peak_angle_deg = float(
        result.angles_deg[peak_index]
    )
    assert peak_angle_deg == pytest.approx(
        0.0,
        abs=0.1,
    )
    assert result.magnitude[peak_index] == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert result.magnitude_db[peak_index] == pytest.approx(
        0.0,
        abs=1e-12,
    )
@pytest.mark.parametrize(
    "steering_angle_deg",
    [
        -60.0,
        -35.0,
        -10.0,
        15.0,
        30.0,
        55.0,
    ],
)
def test_internal_steering_tracks_requested_angle(
    steering_angle_deg: float,
) -> None:
    """Internally steered patterns should peak near the requested angle."""
    result = calculate_ula_array_factor(
        number_of_elements=32,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        steering_angle_deg=steering_angle_deg,
    )
    peak_angle_deg = float(
        result.angles_deg[
            np.argmax(
                result.magnitude
            )
        ]
    )
    assert peak_angle_deg == pytest.approx(
        steering_angle_deg,
        abs=0.1,
    )
def test_complex_steering_weights_match_internal_steering() -> None:
    """Explicit steering weights should reproduce internal steering."""
    steering_angle_deg = 27.0
    number_of_elements = 24
    internal_result = calculate_ula_array_factor(
        number_of_elements=number_of_elements,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        steering_angle_deg=steering_angle_deg,
        weights_include_steering=False,
    )
    steering = generate_steering_weights(
        number_of_elements=number_of_elements,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        steering_angle_deg=steering_angle_deg,
    )
    explicit_result = calculate_ula_array_factor(
        number_of_elements=number_of_elements,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        steering_angle_deg=0.0,
        weights=steering.complex_weights,
        weights_include_steering=True,
    )
    np.testing.assert_allclose(
        explicit_result.magnitude,
        internal_result.magnitude,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        explicit_result.magnitude_db,
        internal_result.magnitude_db,
        rtol=1e-10,
        atol=1e-10,
    )
def test_global_complex_weight_phase_does_not_change_pattern() -> None:
    """A common phase rotation should not change normalized magnitude."""
    number_of_elements = 16
    reference_weights = np.ones(
        number_of_elements,
        dtype=np.complex128,
    )
    rotated_weights = (
        reference_weights
        * np.exp(
            1j * np.deg2rad(73.0)
        )
    )
    reference_result = calculate_ula_array_factor(
        number_of_elements=number_of_elements,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        weights=reference_weights,
    )
    rotated_result = calculate_ula_array_factor(
        number_of_elements=number_of_elements,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        weights=rotated_weights,
    )
    np.testing.assert_allclose(
        rotated_result.magnitude,
        reference_result.magnitude,
        atol=1e-12,
    )
def test_array_factor_is_symmetric_at_broadside() -> None:
    """A centered broadside ULA magnitude pattern should be symmetric."""
    result = calculate_ula_array_factor(
        number_of_elements=15,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        steering_angle_deg=0.0,
    )
    np.testing.assert_allclose(
        result.magnitude,
        result.magnitude[::-1],
        rtol=1e-12,
        atol=1e-12,
    )
def test_increasing_element_count_narrows_main_beam() -> None:
    """A larger array aperture should have a narrower -3 dB beam."""
    small_result = calculate_ula_array_factor(
        number_of_elements=8,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
    )
    large_result = calculate_ula_array_factor(
        number_of_elements=32,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
    )
    small_indices = np.flatnonzero(
        small_result.magnitude_db >= -3.0
    )
    large_indices = np.flatnonzero(
        large_result.magnitude_db >= -3.0
    )
    small_width_deg = float(
        small_result.angles_deg[
            small_indices[-1]
        ]
        - small_result.angles_deg[
            small_indices[0]
        ]
    )
    large_width_deg = float(
        large_result.angles_deg[
            large_indices[-1]
        ]
        - large_result.angles_deg[
            large_indices[0]
        ]
    )
    assert large_width_deg < small_width_deg
def test_half_wavelength_spacing_has_no_equal_height_grating_lobe() -> None:
    """A half-wavelength broadside ULA should have one global maximum."""
    result = calculate_ula_array_factor(
        number_of_elements=16,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
    )
    near_peak_indices = np.flatnonzero(
        result.magnitude_db > -0.01
    )
    near_peak_angles = result.angles_deg[
        near_peak_indices
    ]
    assert np.all(
        np.abs(
            near_peak_angles
        )
        < 1.0
    )
def test_wavelength_and_wave_number_are_returned_correctly() -> None:
    """Returned propagation constants should match analytical values."""
    result = calculate_ula_array_factor(
        number_of_elements=4,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
    )
    expected_wavelength_m = (
        SPEED_OF_LIGHT_M_S
        / FREQUENCY_HZ
    )
    expected_wave_number = (
        2.0
        * np.pi
        / expected_wavelength_m
    )
    assert result.wavelength_m == pytest.approx(
        expected_wavelength_m
    )
    assert result.wave_number_rad_m == pytest.approx(
        expected_wave_number
    )
def test_custom_angle_grid_is_preserved() -> None:
    """The result should retain a valid custom observation grid."""
    custom_angles = np.linspace(
        -45.0,
        45.0,
        901,
    )
    result = calculate_ula_array_factor(
        number_of_elements=12,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        angles_deg=custom_angles,
    )
    np.testing.assert_array_equal(
        result.angles_deg,
        custom_angles,
    )
    assert result.array_factor.shape == custom_angles.shape
    assert result.magnitude.shape == custom_angles.shape
    assert result.magnitude_db.shape == custom_angles.shape
def test_db_floor_is_enforced() -> None:
    """No normalized dB value should fall below the selected floor."""
    minimum_db = -50.0
    result = calculate_ula_array_factor(
        number_of_elements=16,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        minimum_db=minimum_db,
    )
    assert np.min(
        result.magnitude_db
    ) >= minimum_db
@pytest.mark.parametrize(
    ("keyword_arguments", "error_type"),
    [
        (
            {
                "number_of_elements": 1,
                "frequency_hz": FREQUENCY_HZ,
                "element_spacing_m": ELEMENT_SPACING_M,
            },
            ValueError,
        ),
        (
            {
                "number_of_elements": 8,
                "frequency_hz": 0.0,
                "element_spacing_m": ELEMENT_SPACING_M,
            },
            ValueError,
        ),
        (
            {
                "number_of_elements": 8,
                "frequency_hz": FREQUENCY_HZ,
                "element_spacing_m": 0.0,
            },
            ValueError,
        ),
        (
            {
                "number_of_elements": 8,
                "frequency_hz": FREQUENCY_HZ,
                "element_spacing_m": ELEMENT_SPACING_M,
                "steering_angle_deg": 95.0,
            },
            ValueError,
        ),
        (
            {
                "number_of_elements": 8,
                "frequency_hz": FREQUENCY_HZ,
                "element_spacing_m": ELEMENT_SPACING_M,
                "minimum_db": 0.0,
            },
            ValueError,
        ),
    ],
)
def test_invalid_scalar_inputs_raise_errors(
    keyword_arguments: dict[str, float | int],
    error_type: type[Exception],
) -> None:
    """Invalid scalar parameters should fail clearly."""
    with pytest.raises(error_type):
        calculate_ula_array_factor(
            **keyword_arguments,
        )
def test_invalid_weight_length_raises_error() -> None:
    """Weight count must match the number of elements."""
    with pytest.raises(
        ValueError,
        match="weights must have shape",
    ):
        calculate_ula_array_factor(
            number_of_elements=8,
            frequency_hz=FREQUENCY_HZ,
            element_spacing_m=ELEMENT_SPACING_M,
            weights=np.ones(
                7,
                dtype=np.complex128,
            ),
        )
def test_all_zero_weights_raise_error() -> None:
    """An array with no active excitation should be rejected."""
    with pytest.raises(
        ValueError,
        match="only zeros",
    ):
        calculate_ula_array_factor(
            number_of_elements=8,
            frequency_hz=FREQUENCY_HZ,
            element_spacing_m=ELEMENT_SPACING_M,
            weights=np.zeros(
                8,
                dtype=np.complex128,
            ),
        )
@pytest.mark.parametrize(
    "invalid_angles",
    [
        np.array(
            [-10.0, 0.0, -5.0]
        ),
        np.array(
            [-95.0, 0.0, 10.0]
        ),
        np.array(
            [0.0]
        ),
        np.array(
            [[-10.0, 0.0, 10.0]]
        ),
        np.array(
            [-10.0, np.nan, 10.0]
        ),
    ],
)
def test_invalid_angle_grids_raise_errors(
    invalid_angles: np.ndarray,
) -> None:
    """Malformed angle grids should be rejected."""
    with pytest.raises(ValueError):
        calculate_ula_array_factor(
            number_of_elements=8,
            frequency_hz=FREQUENCY_HZ,
            element_spacing_m=ELEMENT_SPACING_M,
            angles_deg=invalid_angles,
        )dx 