"""Tests for radiation-pattern metric extraction."""
from __future__ import annotations
import numpy as np
import pytest
from src.beamformer import (
    SPEED_OF_LIGHT_M_S,
    calculate_ula_array_factor,
)
from src.metrics import (
    analyze_beam_pattern,
    calculate_half_power_beamwidth,
    calculate_peak_sidelobe,
    find_first_null_indices,
)
from src.windows import (
    WindowType,
    generate_window,
)
FREQUENCY_HZ = 28e9
WAVELENGTH_M = SPEED_OF_LIGHT_M_S / FREQUENCY_HZ
ELEMENT_SPACING_M = WAVELENGTH_M / 2.0
def test_uniform_array_metrics_are_physically_reasonable() -> None:
    """A 16-element uniform ULA should produce expected basic metrics."""
    result = calculate_ula_array_factor(
        number_of_elements=16,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
    )
    metrics = analyze_beam_pattern(
        angles_deg=result.angles_deg,
        magnitude_linear=result.magnitude,
        magnitude_db=result.magnitude_db,
        requested_steering_angle_deg=0.0,
    )
    assert metrics.actual_peak_angle_deg == pytest.approx(
        0.0,
        abs=0.1,
    )
    assert metrics.steering_error_deg == pytest.approx(
        0.0,
        abs=0.1,
    )
    assert metrics.half_power_beamwidth_deg is not None
    assert 5.0 < metrics.half_power_beamwidth_deg < 8.0
    assert metrics.first_null_beamwidth_deg is not None
    assert 13.0 < metrics.first_null_beamwidth_deg < 16.0
    assert metrics.peak_sidelobe_level_db is not None
    assert metrics.peak_sidelobe_level_db == pytest.approx(
        -13.15,
        abs=0.8,
    )
    assert not metrics.grating_lobe_detected
def test_detected_peak_tracks_steering_angle() -> None:
    """Metric extraction should identify a steered main beam."""
    result = calculate_ula_array_factor(
        number_of_elements=24,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        steering_angle_deg=32.0,
    )
    metrics = analyze_beam_pattern(
        angles_deg=result.angles_deg,
        magnitude_linear=result.magnitude,
        magnitude_db=result.magnitude_db,
        requested_steering_angle_deg=32.0,
    )
    assert metrics.actual_peak_angle_deg == pytest.approx(
        32.0,
        abs=0.1,
    )
    assert abs(
        metrics.steering_error_deg
    ) <= 0.1
def test_taper_reduces_peak_sidelobe() -> None:
    """Taylor taper should lower sidelobes relative to uniform weights."""
    uniform_result = calculate_ula_array_factor(
        number_of_elements=32,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        weights=generate_window(
            32,
            WindowType.UNIFORM,
        ),
    )
    taylor_result = calculate_ula_array_factor(
        number_of_elements=32,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        weights=generate_window(
            32,
            WindowType.TAYLOR,
            sidelobe_level_db=30.0,
        ),
    )
    uniform_metrics = analyze_beam_pattern(
        uniform_result.angles_deg,
        uniform_result.magnitude,
        uniform_result.magnitude_db,
        requested_steering_angle_deg=0.0,
    )
    taylor_metrics = analyze_beam_pattern(
        taylor_result.angles_deg,
        taylor_result.magnitude,
        taylor_result.magnitude_db,
        requested_steering_angle_deg=0.0,
    )
    assert uniform_metrics.peak_sidelobe_level_db is not None
    assert taylor_metrics.peak_sidelobe_level_db is not None
    assert (
        taylor_metrics.peak_sidelobe_level_db
        < uniform_metrics.peak_sidelobe_level_db
    )
def test_taper_increases_half_power_beamwidth() -> None:
    """Sidelobe suppression should broaden the main beam."""
    uniform_weights = generate_window(
        32,
        WindowType.UNIFORM,
    )
    blackman_weights = generate_window(
        32,
        WindowType.BLACKMAN,
    )
    uniform_result = calculate_ula_array_factor(
        number_of_elements=32,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        weights=uniform_weights,
    )
    blackman_result = calculate_ula_array_factor(
        number_of_elements=32,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        weights=blackman_weights,
    )
    uniform_metrics = analyze_beam_pattern(
        uniform_result.angles_deg,
        uniform_result.magnitude,
        uniform_result.magnitude_db,
        requested_steering_angle_deg=0.0,
    )
    blackman_metrics = analyze_beam_pattern(
        blackman_result.angles_deg,
        blackman_result.magnitude,
        blackman_result.magnitude_db,
        requested_steering_angle_deg=0.0,
    )
    assert uniform_metrics.half_power_beamwidth_deg is not None
    assert blackman_metrics.half_power_beamwidth_deg is not None
    assert (
        blackman_metrics.half_power_beamwidth_deg
        > uniform_metrics.half_power_beamwidth_deg
    )
def test_first_null_indices_surround_main_peak() -> None:
    """The first detected nulls should lie on either side of the peak."""
    result = calculate_ula_array_factor(
        number_of_elements=16,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
    )
    peak_index = int(
        np.argmax(
            result.magnitude
        )
    )
    left_index, right_index = find_first_null_indices(
        magnitude_linear=result.magnitude,
        main_peak_index=peak_index,
    )
    assert left_index is not None
    assert right_index is not None
    assert left_index < peak_index < right_index
    assert result.magnitude[left_index] < 0.02
    assert result.magnitude[right_index] < 0.02
def test_half_power_beamwidth_interpolation() -> None:
    """Threshold interpolation should recover a known synthetic width."""
    angles_deg = np.array(
        [-4.0, -2.0, 0.0, 2.0, 4.0],
        dtype=np.float64,
    )
    magnitude_db = np.array(
        [-8.0, -2.0, 0.0, -2.0, -8.0],
        dtype=np.float64,
    )
    beamwidth = calculate_half_power_beamwidth(
        angles_deg=angles_deg,
        magnitude_db=magnitude_db,
        main_peak_index=2,
        threshold_db=-3.0,
    )
    assert beamwidth == pytest.approx(
        4.6666667,
        abs=1e-6,
    )
def test_peak_sidelobe_excludes_main_beam_region() -> None:
    """The main beam must not be reported as a sidelobe."""
    angles_deg = np.arange(
        -5.0,
        6.0,
        1.0,
    )
    magnitude_db = np.array(
        [
            -40.0,
            -20.0,
            -30.0,
            -15.0,
            -4.0,
            0.0,
            -4.0,
            -18.0,
            -25.0,
            -12.0,
            -35.0,
        ]
    )
    sidelobe_level, sidelobe_angle = calculate_peak_sidelobe(
        angles_deg=angles_deg,
        magnitude_db=magnitude_db,
        left_null_index=3,
        right_null_index=7,
    )
    assert sidelobe_level == pytest.approx(
        -12.0
    )
    assert sidelobe_angle == pytest.approx(
        4.0
    )
def test_large_spacing_triggers_grating_lobe_warning() -> None:
    """Spacing above one wavelength should produce competing major lobes."""
    result = calculate_ula_array_factor(
        number_of_elements=16,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=1.2 * WAVELENGTH_M,
        steering_angle_deg=0.0,
    )
    metrics = analyze_beam_pattern(
        angles_deg=result.angles_deg,
        magnitude_linear=result.magnitude,
        magnitude_db=result.magnitude_db,
        requested_steering_angle_deg=0.0,
        major_lobe_threshold_db=-1.0,
    )
    assert metrics.grating_lobe_detected
    assert metrics.number_of_major_lobes > 1
    assert any(
        "grating lobes" in warning.lower()
        for warning in metrics.warnings
    )
def test_steering_error_warning_is_created() -> None:
    """A deliberately wrong requested angle should create a warning."""
    result = calculate_ula_array_factor(
        number_of_elements=16,
        frequency_hz=FREQUENCY_HZ,
        element_spacing_m=ELEMENT_SPACING_M,
        steering_angle_deg=0.0,
    )
    metrics = analyze_beam_pattern(
        angles_deg=result.angles_deg,
        magnitude_linear=result.magnitude,
        magnitude_db=result.magnitude_db,
        requested_steering_angle_deg=20.0,
        steering_tolerance_deg=1.0,
    )
    assert abs(
        metrics.steering_error_deg
    ) > 1.0
    assert any(
        "differs from the requested" in warning
        for warning in metrics.warnings
    )
def test_invalid_metric_shapes_raise_error() -> None:
    """Pattern arrays must have matching one-dimensional shapes."""
    with pytest.raises(ValueError):
        analyze_beam_pattern(
            angles_deg=np.linspace(
                -90.0,
                90.0,
                101,
            ),
            magnitude_linear=np.ones(
                100
            ),
            magnitude_db=np.ones(
                101
            ),
            requested_steering_angle_deg=0.0,
        )
def test_negative_linear_magnitude_raises_error() -> None:
    """Linear magnitude values cannot be negative."""
    angles = np.linspace(
        -10.0,
        10.0,
        11,
    )
    magnitude = np.ones(
        11
    )
    magnitude[3] = -0.1
    with pytest.raises(
        ValueError,
        match="negative",
    ):
        analyze_beam_pattern(
            angles_deg=angles,
            magnitude_linear=magnitude,
            magnitude_db=np.zeros(
                11
            ),
            requested_steering_angle_deg=0.0,
        )