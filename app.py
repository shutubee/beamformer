from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.adaptive_beamforming import (
    calculate_beamformer_response,
    estimate_sample_covariance,
    mvdr_weights,
    steering_vector_ula,
)
from src.beamformer import (
    SPEED_OF_LIGHT_M_S,
    calculate_ula_array_factor,
)
from src.calibration import (
    calculate_correction_weights,
    estimate_reference_response,
    simulate_calibration_measurement,
    validate_calibration,
)
from src.direction_finding import (
    calculate_bartlett_spectrum,
    calculate_capon_spectrum,
    calculate_music_spectrum,
    estimate_source_count_mdl,
)
from src.imperfections import apply_array_imperfections
from src.metrics import analyze_beam_pattern
from src.signal_simulation import (
    SourceDefinition,
    simulate_ula_snapshots,
)
from src.steering import generate_steering_weights
from src.wideband_beamforming import (
    analyze_beam_squint,
    calculate_wideband_phase_shifter_response,
    calculate_wideband_ttd_response,
)
from src.windows import WindowType, generate_window


st.set_page_config(
    page_title="Array Beamformer Studio",
    page_icon="📡",
    layout="wide",
)


def plot_pattern(
    angles_deg: np.ndarray,
    magnitude_db: np.ndarray,
    *,
    title: str,
    traces: list[tuple[np.ndarray, str]] | None = None,
    steering_angle_deg: float | None = None,
    minimum_db: float = -70.0,
) -> go.Figure:
    """Create a Cartesian radiation-pattern plot."""

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=angles_deg,
            y=magnitude_db,
            mode="lines",
            name="Pattern",
        )
    )

    if traces:
        figure.data = ()

        for trace_values, trace_name in traces:
            figure.add_trace(
                go.Scatter(
                    x=angles_deg,
                    y=trace_values,
                    mode="lines",
                    name=trace_name,
                )
            )

    if steering_angle_deg is not None:
        figure.add_vline(
            x=steering_angle_deg,
            line_dash="dash",
            annotation_text="Requested steering",
        )

    figure.update_layout(
        title=title,
        xaxis_title="Observation angle (degrees)",
        yaxis_title="Normalized magnitude (dB)",
        xaxis_range=[-90.0, 90.0],
        yaxis_range=[minimum_db, 0.0],
        hovermode="x unified",
        legend_title="Response",
    )

    return figure


def plot_polar_pattern(
    angles_deg: np.ndarray,
    magnitude_db: np.ndarray,
    *,
    title: str,
    minimum_db: float = -60.0,
) -> go.Figure:
    """Create a polar radiation-pattern plot."""

    radial_values = magnitude_db - minimum_db

    radial_values = np.maximum(
        radial_values,
        0.0,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatterpolar(
            theta=angles_deg,
            r=radial_values,
            mode="lines",
            name="Array factor",
        )
    )

    figure.update_layout(
        title=title,
        polar={
            "radialaxis": {
                "visible": True,
                "range": [
                    0.0,
                    abs(minimum_db),
                ],
                "tickvals": [
                    0.0,
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                ],
                "ticktext": [
                    f"{minimum_db:.0f}",
                    f"{minimum_db + 10:.0f}",
                    f"{minimum_db + 20:.0f}",
                    f"{minimum_db + 30:.0f}",
                    f"{minimum_db + 40:.0f}",
                    f"{minimum_db + 50:.0f}",
                    "0",
                ],
            },
            "angularaxis": {
                "direction": "counterclockwise",
                "rotation": 90,
            },
        },
    )

    return figure


def format_optional(
    value: float | None,
    unit: str,
    *,
    digits: int = 2,
) -> str:
    """Format an optional numerical metric."""

    if value is None or not np.isfinite(value):
        return "N/A"

    return f"{value:.{digits}f} {unit}"


st.title("📡 Array Beamformer Studio")

st.write(
    "Interactive phased-array simulation for beam steering, "
    "sidelobe control, hardware imperfections, adaptive beamforming, "
    "direction finding, wideband beam squint, and calibration."
)

with st.sidebar:
    st.header("Common array settings")

    frequency_ghz = st.number_input(
        "Operating frequency (GHz)",
        min_value=0.1,
        max_value=300.0,
        value=28.0,
        step=0.5,
    )

    frequency_hz = frequency_ghz * 1e9
    wavelength_m = SPEED_OF_LIGHT_M_S / frequency_hz

    number_of_elements = st.slider(
        "Number of elements",
        min_value=2,
        max_value=128,
        value=16,
        step=1,
    )

    spacing_wavelengths = st.slider(
        "Element spacing (λ)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.05,
    )

    element_spacing_m = (
        spacing_wavelengths
        * wavelength_m
    )

    steering_angle_deg = st.slider(
        "Steering angle (degrees)",
        min_value=-80.0,
        max_value=80.0,
        value=20.0,
        step=1.0,
    )

    minimum_db = st.slider(
        "Plot floor (dB)",
        min_value=-120,
        max_value=-30,
        value=-70,
        step=5,
    )

    st.divider()

    st.metric(
        "Wavelength",
        f"{wavelength_m * 1e3:.3f} mm",
    )

    st.metric(
        "Physical spacing",
        f"{element_spacing_m * 1e3:.3f} mm",
    )

    array_length_m = (
        number_of_elements - 1
    ) * element_spacing_m

    st.metric(
        "Array length",
        f"{array_length_m * 1e3:.2f} mm",
    )


(
    basic_tab,
    windows_tab,
    imperfections_tab,
    adaptive_tab,
    doa_tab,
    wideband_tab,
    calibration_tab,
) = st.tabs(
    [
        "Basic beam",
        "Window comparison",
        "Imperfections",
        "MVDR",
        "Direction finding",
        "Wideband",
        "Calibration",
    ]
)


with basic_tab:
    st.subheader("Uniform linear array beam steering")

    control_column, output_column = st.columns(
        [1, 2]
    )

    with control_column:
        selected_window_name = st.selectbox(
            "Amplitude window",
            options=[
                item.value
                for item in WindowType
            ],
            index=0,
            key="basic_window",
        )

        taylor_sll_db = st.slider(
            "Taylor sidelobe target (dB)",
            min_value=20.0,
            max_value=60.0,
            value=30.0,
            step=1.0,
        )

        chebyshev_attenuation_db = st.slider(
            "Chebyshev attenuation (dB)",
            min_value=30.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
        )

        use_practical_phase = st.checkbox(
            "Use practical phase shifter",
            value=False,
        )

        phase_bits: int | None = None
        phase_error_std_deg = 0.0

        if use_practical_phase:
            phase_bits = st.slider(
                "Phase-shifter resolution (bits)",
                min_value=1,
                max_value=12,
                value=4,
            )

            phase_error_std_deg = st.slider(
                "Random phase error σ (degrees)",
                min_value=0.0,
                max_value=20.0,
                value=2.0,
                step=0.5,
            )

    amplitude_window = generate_window(
        number_of_elements=number_of_elements,
        window_type=selected_window_name,
        taylor_nbar=min(
            4,
            number_of_elements,
        ),
        sidelobe_level_db=taylor_sll_db,
        chebyshev_attenuation_db=(
            chebyshev_attenuation_db
        ),
    )

    if use_practical_phase:
        steering = generate_steering_weights(
            number_of_elements=number_of_elements,
            frequency_hz=frequency_hz,
            element_spacing_m=element_spacing_m,
            steering_angle_deg=steering_angle_deg,
            amplitude_weights=np.abs(
                amplitude_window
            ),
            quantization_bits=phase_bits,
            phase_error_std_deg=phase_error_std_deg,
            seed=42,
        )

        basic_result = calculate_ula_array_factor(
            number_of_elements=number_of_elements,
            frequency_hz=frequency_hz,
            element_spacing_m=element_spacing_m,
            steering_angle_deg=0.0,
            weights=steering.complex_weights,
            weights_include_steering=True,
            minimum_db=float(
                minimum_db
            ),
        )
    else:
        basic_result = calculate_ula_array_factor(
            number_of_elements=number_of_elements,
            frequency_hz=frequency_hz,
            element_spacing_m=element_spacing_m,
            steering_angle_deg=steering_angle_deg,
            weights=amplitude_window,
            weights_include_steering=False,
            minimum_db=float(
                minimum_db
            ),
        )

    basic_metrics = analyze_beam_pattern(
        angles_deg=basic_result.angles_deg,
        magnitude_linear=basic_result.magnitude,
        magnitude_db=basic_result.magnitude_db,
        requested_steering_angle_deg=(
            steering_angle_deg
        ),
    )

    with output_column:
        metric_columns = st.columns(5)

        metric_columns[0].metric(
            "Detected peak",
            f"{basic_metrics.actual_peak_angle_deg:.2f}°",
        )

        metric_columns[1].metric(
            "Steering error",
            f"{basic_metrics.steering_error_deg:.2f}°",
        )

        metric_columns[2].metric(
            "HPBW",
            format_optional(
                basic_metrics.half_power_beamwidth_deg,
                "°",
            ),
        )

        metric_columns[3].metric(
            "Peak sidelobe",
            format_optional(
                basic_metrics.peak_sidelobe_level_db,
                "dB",
            ),
        )

        metric_columns[4].metric(
            "Grating lobes",
            "Detected"
            if basic_metrics.grating_lobe_detected
            else "None",
        )

    chart_column, polar_column = st.columns(2)

    with chart_column:
        st.plotly_chart(
            plot_pattern(
                basic_result.angles_deg,
                basic_result.magnitude_db,
                title="Normalized ULA pattern",
                steering_angle_deg=(
                    steering_angle_deg
                ),
                minimum_db=float(
                    minimum_db
                ),
            ),
            use_container_width=True,
        )

    with polar_column:
        st.plotly_chart(
            plot_polar_pattern(
                basic_result.angles_deg,
                basic_result.magnitude_db,
                title="Polar radiation pattern",
                minimum_db=float(
                    minimum_db
                ),
            ),
            use_container_width=True,
        )

    element_table = pd.DataFrame(
        {
            "Element": np.arange(
                number_of_elements
            ),
            "Amplitude": np.abs(
                amplitude_window
            ),
        }
    )

    if use_practical_phase:
        element_table[
            "Ideal phase (deg)"
        ] = np.rad2deg(
            steering.ideal_phase_rad
        )

        element_table[
            "Applied phase (deg)"
        ] = np.rad2deg(
            steering.applied_phase_rad
        )

        element_table[
            "Phase error (deg)"
        ] = np.rad2deg(
            steering.phase_error_rad
        )

    with st.expander(
        "Element excitation table"
    ):
        st.dataframe(
            element_table,
            use_container_width=True,
            hide_index=True,
        )

    for warning in basic_metrics.warnings:
        st.warning(warning)


with windows_tab:
    st.subheader(
        "Amplitude-window comparison"
    )

    selected_windows = st.multiselect(
        "Select windows",
        options=[
            item.value
            for item in WindowType
        ],
        default=[
            WindowType.UNIFORM.value,
            WindowType.HAMMING.value,
            WindowType.TAYLOR.value,
            WindowType.DOLPH_CHEBYSHEV.value,
        ],
    )

    if not selected_windows:
        st.info(
            "Select at least one amplitude window."
        )
    else:
        traces: list[
            tuple[np.ndarray, str]
        ] = []

        window_rows: list[
            dict[str, float | str | bool]
        ] = []

        for window_name in selected_windows:
            window_weights = generate_window(
                number_of_elements=number_of_elements,
                window_type=window_name,
                taylor_nbar=min(
                    4,
                    number_of_elements,
                ),
                sidelobe_level_db=30.0,
                chebyshev_attenuation_db=50.0,
            )

            window_result = (
                calculate_ula_array_factor(
                    number_of_elements=(
                        number_of_elements
                    ),
                    frequency_hz=frequency_hz,
                    element_spacing_m=(
                        element_spacing_m
                    ),
                    steering_angle_deg=(
                        steering_angle_deg
                    ),
                    weights=window_weights,
                    minimum_db=float(
                        minimum_db
                    ),
                )
            )

            window_metrics = analyze_beam_pattern(
                angles_deg=(
                    window_result.angles_deg
                ),
                magnitude_linear=(
                    window_result.magnitude
                ),
                magnitude_db=(
                    window_result.magnitude_db
                ),
                requested_steering_angle_deg=(
                    steering_angle_deg
                ),
            )

            traces.append(
                (
                    window_result.magnitude_db,
                    window_name,
                )
            )

            window_rows.append(
                {
                    "Window": window_name,
                    "Peak angle (deg)": (
                        window_metrics.actual_peak_angle_deg
                    ),
                    "HPBW (deg)": (
                        window_metrics.half_power_beamwidth_deg
                    ),
                    "FNBW (deg)": (
                        window_metrics.first_null_beamwidth_deg
                    ),
                    "Peak sidelobe (dB)": (
                        window_metrics.peak_sidelobe_level_db
                    ),
                    "Grating lobe": (
                        window_metrics.grating_lobe_detected
                    ),
                }
            )

        st.plotly_chart(
            plot_pattern(
                window_result.angles_deg,
                window_result.magnitude_db,
                title="Window comparison",
                traces=traces,
                steering_angle_deg=(
                    steering_angle_deg
                ),
                minimum_db=float(
                    minimum_db
                ),
            ),
            use_container_width=True,
        )

        st.dataframe(
            pd.DataFrame(
                window_rows
            ),
            use_container_width=True,
            hide_index=True,
        )


with imperfections_tab:
    st.subheader(
        "Hardware imperfections"
    )

    settings_column, plot_column = st.columns(
        [1, 2]
    )

    with settings_column:
        imperfection_window = st.selectbox(
            "Base amplitude window",
            options=[
                item.value
                for item in WindowType
            ],
            index=4,
            key="imperfection_window",
        )

        gain_error_std_db = st.slider(
            "Gain mismatch σ (dB)",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.1,
        )

        phase_error_std_deg = st.slider(
            "Phase error σ (degrees)",
            min_value=0.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            key="imperfection_phase_error",
        )

        failure_probability = st.slider(
            "Random failure probability",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
        )

        use_phase_quantization = st.checkbox(
            "Apply phase quantization",
            value=True,
        )

        imperfection_bits: int | None = None

        if use_phase_quantization:
            imperfection_bits = st.slider(
                "Quantization bits",
                min_value=1,
                max_value=12,
                value=4,
                key="imperfection_bits",
            )

    base_window = generate_window(
        number_of_elements=number_of_elements,
        window_type=imperfection_window,
        taylor_nbar=min(
            4,
            number_of_elements,
        ),
        sidelobe_level_db=30.0,
    )

    ideal_steering = generate_steering_weights(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        steering_angle_deg=steering_angle_deg,
        amplitude_weights=np.abs(
            base_window
        ),
    )

    imperfect_weights = apply_array_imperfections(
        ideal_weights=(
            ideal_steering.complex_weights
        ),
        gain_error_std_db=gain_error_std_db,
        phase_error_std_deg=(
            phase_error_std_deg
        ),
        random_failure_probability=(
            failure_probability
        ),
        quantization_bits=imperfection_bits,
        seed=42,
    )

    ideal_pattern = calculate_ula_array_factor(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        weights=ideal_steering.complex_weights,
        weights_include_steering=True,
        minimum_db=float(
            minimum_db
        ),
    )

    imperfect_pattern = calculate_ula_array_factor(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        weights=(
            imperfect_weights.imperfect_weights
        ),
        weights_include_steering=True,
        minimum_db=float(
            minimum_db
        ),
    )

    with plot_column:
        st.plotly_chart(
            plot_pattern(
                ideal_pattern.angles_deg,
                ideal_pattern.magnitude_db,
                title="Ideal versus imperfect array",
                traces=[
                    (
                        ideal_pattern.magnitude_db,
                        "Ideal",
                    ),
                    (
                        imperfect_pattern.magnitude_db,
                        "Imperfect",
                    ),
                ],
                steering_angle_deg=(
                    steering_angle_deg
                ),
                minimum_db=float(
                    minimum_db
                ),
            ),
            use_container_width=True,
        )

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Active elements",
        imperfect_weights.number_of_active_elements,
    )

    metric_columns[1].metric(
        "Failed elements",
        imperfect_weights.failed_element_indices.size,
    )

    metric_columns[2].metric(
        "Failure fraction",
        f"{100.0 * imperfect_weights.failure_fraction:.1f}%",
    )

    imperfect_metrics = analyze_beam_pattern(
        angles_deg=imperfect_pattern.angles_deg,
        magnitude_linear=(
            imperfect_pattern.magnitude
        ),
        magnitude_db=(
            imperfect_pattern.magnitude_db
        ),
        requested_steering_angle_deg=(
            steering_angle_deg
        ),
    )

    metric_columns[3].metric(
        "Imperfect peak angle",
        f"{imperfect_metrics.actual_peak_angle_deg:.2f}°",
    )


with adaptive_tab:
    st.subheader(
        "MVDR adaptive beamforming"
    )

    adaptive_controls, adaptive_output = st.columns(
        [1, 2]
    )

    with adaptive_controls:
        desired_angle = st.slider(
            "Desired source angle",
            min_value=-70.0,
            max_value=70.0,
            value=20.0,
            step=1.0,
        )

        interferer_one = st.slider(
            "Interferer 1 angle",
            min_value=-80.0,
            max_value=80.0,
            value=-30.0,
            step=1.0,
        )

        interferer_two = st.slider(
            "Interferer 2 angle",
            min_value=-80.0,
            max_value=80.0,
            value=50.0,
            step=1.0,
        )

        desired_power_db = st.slider(
            "Desired source power (dB)",
            min_value=-10.0,
            max_value=30.0,
            value=10.0,
            step=1.0,
        )

        interference_power_db = st.slider(
            "Interference power (dB)",
            min_value=0.0,
            max_value=40.0,
            value=22.0,
            step=1.0,
        )

        number_of_snapshots = st.slider(
            "Snapshots",
            min_value=100,
            max_value=10000,
            value=2000,
            step=100,
        )

        diagonal_loading = st.number_input(
            "Diagonal loading",
            min_value=0.0,
            max_value=1.0,
            value=0.001,
            step=0.001,
            format="%.4f",
        )

    adaptive_simulation = simulate_ula_snapshots(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        number_of_snapshots=number_of_snapshots,
        sources=[
            SourceDefinition(
                angle_deg=desired_angle,
                power_db=desired_power_db,
                label="Desired",
                signal_type="qpsk",
            ),
            SourceDefinition(
                angle_deg=interferer_one,
                power_db=interference_power_db,
                label="Interferer 1",
            ),
            SourceDefinition(
                angle_deg=interferer_two,
                power_db=interference_power_db,
                label="Interferer 2",
            ),
        ],
        noise_power_db=0.0,
        seed=42,
    )

    covariance = estimate_sample_covariance(
        snapshots=adaptive_simulation.snapshots,
        diagonal_loading=diagonal_loading,
    )

    desired_vector = steering_vector_ula(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        angle_deg=desired_angle,
    )

    adaptive_weights = mvdr_weights(
        covariance_matrix=(
            covariance.covariance_matrix
        ),
        steering_vector=desired_vector,
        diagonal_loading=diagonal_loading,
        use_pseudoinverse=True,
    )

    scan_angles = np.linspace(
        -90.0,
        90.0,
        1801,
    )

    scan_vectors = np.column_stack(
        [
            steering_vector_ula(
                number_of_elements=(
                    number_of_elements
                ),
                frequency_hz=frequency_hz,
                element_spacing_m=(
                    element_spacing_m
                ),
                angle_deg=float(angle),
            )
            for angle in scan_angles
        ]
    )

    adaptive_response = np.abs(
        calculate_beamformer_response(
            weights=adaptive_weights.weights,
            steering_vectors=scan_vectors,
        )
    )

    adaptive_response /= np.max(
        adaptive_response
    )

    adaptive_response_db = (
        20.0
        * np.log10(
            np.maximum(
                adaptive_response,
                10.0 ** (
                    minimum_db / 20.0
                ),
            )
        )
    )

    with adaptive_output:
        adaptive_figure = plot_pattern(
            scan_angles,
            adaptive_response_db,
            title="MVDR spatial response",
            minimum_db=float(
                minimum_db
            ),
        )

        adaptive_figure.add_vline(
            x=desired_angle,
            line_dash="dash",
            annotation_text="Desired",
        )

        adaptive_figure.add_vline(
            x=interferer_one,
            line_dash="dot",
            annotation_text="Interferer 1",
        )

        adaptive_figure.add_vline(
            x=interferer_two,
            line_dash="dot",
            annotation_text="Interferer 2",
        )

        st.plotly_chart(
            adaptive_figure,
            use_container_width=True,
        )

    mvdr_metrics = st.columns(3)

    mvdr_metrics[0].metric(
        "Output power",
        f"{adaptive_weights.output_power:.4g}",
    )

    mvdr_metrics[1].metric(
        "Distortionless response",
        f"{abs(adaptive_weights.distortionless_response):.4f}",
    )

    mvdr_metrics[2].metric(
        "Covariance condition number",
        f"{adaptive_weights.condition_number:.3e}",
    )


with doa_tab:
    st.subheader(
        "Direction-of-arrival estimation"
    )

    doa_controls, doa_output = st.columns(
        [1, 2]
    )

    with doa_controls:
        doa_angle_1 = st.slider(
            "Source 1 angle",
            min_value=-75.0,
            max_value=75.0,
            value=-25.0,
            step=1.0,
        )

        doa_angle_2 = st.slider(
            "Source 2 angle",
            min_value=-75.0,
            max_value=75.0,
            value=10.0,
            step=1.0,
        )

        doa_angle_3 = st.slider(
            "Source 3 angle",
            min_value=-75.0,
            max_value=75.0,
            value=35.0,
            step=1.0,
        )

        doa_snapshots = st.slider(
            "DOA snapshots",
            min_value=100,
            max_value=10000,
            value=3000,
            step=100,
        )

        automatic_source_count = st.checkbox(
            "Estimate source count using MDL",
            value=True,
        )

    doa_simulation = simulate_ula_snapshots(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        number_of_snapshots=doa_snapshots,
        sources=[
            SourceDefinition(
                angle_deg=doa_angle_1,
                power_db=14.0,
                label="Source 1",
            ),
            SourceDefinition(
                angle_deg=doa_angle_2,
                power_db=11.0,
                label="Source 2",
            ),
            SourceDefinition(
                angle_deg=doa_angle_3,
                power_db=17.0,
                label="Source 3",
            ),
        ],
        noise_power_db=0.0,
        seed=42,
    )

    if automatic_source_count:
        model_order = estimate_source_count_mdl(
            covariance_matrix=(
                doa_simulation.sample_covariance
            ),
            number_of_snapshots=doa_snapshots,
            maximum_sources=min(
                6,
                number_of_elements - 1,
            ),
        )

        estimated_source_count = max(
            1,
            model_order.estimated_number_of_sources,
        )
    else:
        estimated_source_count = st.slider(
            "Assumed source count",
            min_value=1,
            max_value=min(
                6,
                number_of_elements - 1,
            ),
            value=min(
                3,
                number_of_elements - 1,
            ),
        )

    bartlett_result = (
        calculate_bartlett_spectrum(
            covariance_matrix=(
                doa_simulation.sample_covariance
            ),
            number_of_elements=(
                number_of_elements
            ),
            frequency_hz=frequency_hz,
            element_spacing_m=(
                element_spacing_m
            ),
            number_of_sources=(
                estimated_source_count
            ),
            minimum_db=float(
                minimum_db
            ),
        )
    )

    capon_result = calculate_capon_spectrum(
        covariance_matrix=(
            doa_simulation.sample_covariance
        ),
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        number_of_sources=estimated_source_count,
        diagonal_loading=1e-3,
        use_pseudoinverse=True,
        minimum_db=float(
            minimum_db
        ),
    )

    music_result = calculate_music_spectrum(
        covariance_matrix=(
            doa_simulation.sample_covariance
        ),
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        number_of_sources=min(
            estimated_source_count,
            number_of_elements - 1,
        ),
        minimum_db=float(
            minimum_db
        ),
    )

    with doa_output:
        doa_figure = plot_pattern(
            music_result.scan_angles_deg,
            music_result.spectrum_db,
            title="Spatial-spectrum comparison",
            traces=[
                (
                    bartlett_result.spectrum_db,
                    "Bartlett",
                ),
                (
                    capon_result.spectrum_db,
                    "Capon",
                ),
                (
                    music_result.spectrum_db,
                    "MUSIC",
                ),
            ],
            minimum_db=float(
                minimum_db
            ),
        )

        for true_angle in [
            doa_angle_1,
            doa_angle_2,
            doa_angle_3,
        ]:
            doa_figure.add_vline(
                x=true_angle,
                line_dash="dot",
            )

        st.plotly_chart(
            doa_figure,
            use_container_width=True,
        )

    doa_metrics = st.columns(3)

    doa_metrics[0].metric(
        "Estimated source count",
        estimated_source_count,
    )

    doa_metrics[1].metric(
        "MUSIC detections",
        ", ".join(
            f"{value:.1f}°"
            for value in (
                music_result.detected_angles_deg
            )
        ),
    )

    doa_metrics[2].metric(
        "Capon detections",
        ", ".join(
            f"{value:.1f}°"
            for value in (
                capon_result.detected_angles_deg
            )
        ),
    )


with wideband_tab:
    st.subheader(
        "Wideband beam squint"
    )

    wideband_controls, wideband_output = st.columns(
        [1, 2]
    )

    with wideband_controls:
        lower_frequency_ghz = st.number_input(
            "Lower frequency (GHz)",
            min_value=0.1,
            max_value=frequency_ghz,
            value=max(
                0.1,
                frequency_ghz * 0.85,
            ),
            step=0.5,
        )

        upper_frequency_ghz = st.number_input(
            "Upper frequency (GHz)",
            min_value=frequency_ghz,
            max_value=500.0,
            value=frequency_ghz * 1.15,
            step=0.5,
        )

        frequency_points = st.slider(
            "Frequency points",
            min_value=3,
            max_value=101,
            value=33,
            step=2,
        )

    wideband_frequencies_hz = np.linspace(
        lower_frequency_ghz * 1e9,
        upper_frequency_ghz * 1e9,
        frequency_points,
    )

    phase_shifter_wideband = (
        calculate_wideband_phase_shifter_response(
            number_of_elements=(
                number_of_elements
            ),
            frequencies_hz=(
                wideband_frequencies_hz
            ),
            reference_frequency_hz=(
                frequency_hz
            ),
            element_spacing_m=(
                element_spacing_m
            ),
            steering_angle_deg=(
                steering_angle_deg
            ),
            minimum_db=float(
                minimum_db
            ),
        )
    )

    ttd_wideband = (
        calculate_wideband_ttd_response(
            number_of_elements=(
                number_of_elements
            ),
            frequencies_hz=(
                wideband_frequencies_hz
            ),
            reference_frequency_hz=(
                frequency_hz
            ),
            element_spacing_m=(
                element_spacing_m
            ),
            steering_angle_deg=(
                steering_angle_deg
            ),
            minimum_db=float(
                minimum_db
            ),
        )
    )

    phase_squint = analyze_beam_squint(
        phase_shifter_wideband
    )

    ttd_squint = analyze_beam_squint(
        ttd_wideband
    )

    with wideband_output:
        squint_figure = go.Figure()

        squint_figure.add_trace(
            go.Scatter(
                x=(
                    wideband_frequencies_hz
                    / 1e9
                ),
                y=(
                    phase_shifter_wideband
                    .peak_angles_deg
                ),
                mode="lines+markers",
                name="Phase shifter",
            )
        )

        squint_figure.add_trace(
            go.Scatter(
                x=(
                    wideband_frequencies_hz
                    / 1e9
                ),
                y=ttd_wideband.peak_angles_deg,
                mode="lines+markers",
                name="True-time delay",
            )
        )

        squint_figure.add_hline(
            y=steering_angle_deg,
            line_dash="dash",
            annotation_text="Requested direction",
        )

        squint_figure.update_layout(
            title="Peak beam direction versus frequency",
            xaxis_title="Frequency (GHz)",
            yaxis_title="Peak angle (degrees)",
            hovermode="x unified",
        )

        st.plotly_chart(
            squint_figure,
            use_container_width=True,
        )

    squint_metrics = st.columns(4)

    squint_metrics[0].metric(
        "Phase-shifter max squint",
        f"{phase_squint.maximum_absolute_squint_deg:.2f}°",
    )

    squint_metrics[1].metric(
        "Phase-shifter RMS squint",
        f"{phase_squint.rms_squint_deg:.2f}°",
    )

    squint_metrics[2].metric(
        "TTD max squint",
        f"{ttd_squint.maximum_absolute_squint_deg:.2f}°",
    )

    squint_metrics[3].metric(
        "Worst frequency",
        f"{phase_squint.worst_frequency_hz / 1e9:.2f} GHz",
    )


with calibration_tab:
    st.subheader(
        "Array gain and phase calibration"
    )

    calibration_controls, calibration_output = st.columns(
        [1, 2]
    )

    with calibration_controls:
        calibration_angle_deg = st.slider(
            "Calibration source angle",
            min_value=-70.0,
            max_value=70.0,
            value=10.0,
            step=1.0,
        )

        calibration_snapshots = st.slider(
            "Calibration snapshots",
            min_value=100,
            max_value=10000,
            value=3000,
            step=100,
        )

        calibration_gain_error_db = st.slider(
            "True gain error σ (dB)",
            min_value=0.0,
            max_value=3.0,
            value=0.8,
            step=0.1,
        )

        calibration_phase_error_deg = st.slider(
            "True phase error σ (degrees)",
            min_value=0.0,
            max_value=30.0,
            value=8.0,
            step=0.5,
        )

        calibration_noise_power = st.number_input(
            "Calibration noise power",
            min_value=0.0,
            max_value=10.0,
            value=0.02,
            step=0.01,
        )

        maximum_correction_db = st.slider(
            "Maximum correction gain (dB)",
            min_value=1.0,
            max_value=30.0,
            value=10.0,
            step=1.0,
        )

    calibration_random_generator = (
        np.random.default_rng(
            42
        )
    )

    reference_signal = (
        calibration_random_generator.normal(
            size=calibration_snapshots
        )
        + 1j
        * calibration_random_generator.normal(
            size=calibration_snapshots
        )
    ) / np.sqrt(2.0)

    calibration_vector = steering_vector_ula(
        number_of_elements=number_of_elements,
        frequency_hz=frequency_hz,
        element_spacing_m=element_spacing_m,
        angle_deg=calibration_angle_deg,
    )

    (
        calibration_measurements,
        true_hardware_response,
    ) = simulate_calibration_measurement(
        ideal_steering_vector=(
            calibration_vector
        ),
        reference_signal=reference_signal,
        gain_error_std_db=(
            calibration_gain_error_db
        ),
        phase_error_std_deg=(
            calibration_phase_error_deg
        ),
        noise_power_linear=(
            calibration_noise_power
        ),
        seed=42,
    )

    calibration_estimate = (
        estimate_reference_response(
            snapshots=(
                calibration_measurements
            ),
            reference_signal=reference_signal,
            ideal_steering_vector=(
                calibration_vector
            ),
            reference_element_index=0,
        )
    )

    correction = calculate_correction_weights(
        calibration_estimate=(
            calibration_estimate
        ),
        maximum_gain_correction_db=(
            maximum_correction_db
        ),
        normalize_mode="reference",
    )

    calibration_validation = validate_calibration(
        measured_hardware_response=(
            true_hardware_response
        ),
        correction_weights=(
            correction
            .normalized_correction_weights
        ),
        reference_element_index=0,
    )

    with calibration_output:
        element_indices = np.arange(
            number_of_elements
        )

        calibration_figure = go.Figure()

        calibration_figure.add_trace(
            go.Scatter(
                x=element_indices,
                y=(
                    calibration_estimate
                    .gain_error_db
                ),
                mode="lines+markers",
                name="Estimated gain error",
            )
        )

        calibration_figure.add_trace(
            go.Scatter(
                x=element_indices,
                y=(
                    calibration_validation
                    .residual_gain_error_db
                ),
                mode="lines+markers",
                name="Residual gain error",
            )
        )

        calibration_figure.update_layout(
            title="Gain calibration",
            xaxis_title="Element index",
            yaxis_title="Relative gain error (dB)",
        )

        st.plotly_chart(
            calibration_figure,
            use_container_width=True,
        )

        phase_calibration_figure = go.Figure()

        phase_calibration_figure.add_trace(
            go.Scatter(
                x=element_indices,
                y=(
                    calibration_estimate
                    .phase_error_deg
                ),
                mode="lines+markers",
                name="Estimated phase error",
            )
        )

        phase_calibration_figure.add_trace(
            go.Scatter(
                x=element_indices,
                y=(
                    calibration_validation
                    .residual_phase_error_deg
                ),
                mode="lines+markers",
                name="Residual phase error",
            )
        )

        phase_calibration_figure.update_layout(
            title="Phase calibration",
            xaxis_title="Element index",
            yaxis_title="Relative phase error (degrees)",
        )

        st.plotly_chart(
            phase_calibration_figure,
            use_container_width=True,
        )

    calibration_metrics = st.columns(4)

    calibration_metrics[0].metric(
        "Residual RMS gain error",
        f"{calibration_validation.rms_gain_error_db:.4f} dB",
    )

    calibration_metrics[1].metric(
        "Maximum gain error",
        f"{calibration_validation.maximum_absolute_gain_error_db:.4f} dB",
    )

    calibration_metrics[2].metric(
        "Residual RMS phase error",
        f"{calibration_validation.rms_phase_error_deg:.4f}°",
    )

    calibration_metrics[3].metric(
        "Maximum phase error",
        f"{calibration_validation.maximum_absolute_phase_error_deg:.4f}°",
    )


st.divider()

st.caption(
    "Array Beamformer Studio — ULA beam steering, adaptive "
    "beamforming, direction finding, wideband response, and "
    "array calibration."
)

