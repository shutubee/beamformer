"""Estimate multiple source directions using MUSIC."""

import matplotlib.pyplot as plt

from src.direction_finding import (
    calculate_bartlett_spectrum,
    calculate_capon_spectrum,
    calculate_direction_errors,
    calculate_music_spectrum,
    estimate_source_count_mdl,
)
from src.signal_simulation import (
    SourceDefinition,
    simulate_ula_snapshots,
)


frequency_hz = 28e9
speed_of_light_m_s = 299_792_458.0
wavelength_m = speed_of_light_m_s / frequency_hz

number_of_elements = 16
number_of_snapshots = 4000

true_angles_deg = [-28.0, 12.0, 38.0]

simulation = simulate_ula_snapshots(
    number_of_elements=number_of_elements,
    frequency_hz=frequency_hz,
    element_spacing_m=wavelength_m / 2.0,
    number_of_snapshots=number_of_snapshots,
    sources=[
        SourceDefinition(
            angle_deg=-28.0,
            power_db=14.0,
            label="Source 1",
        ),
        SourceDefinition(
            angle_deg=12.0,
            power_db=10.0,
            label="Source 2",
        ),
        SourceDefinition(
            angle_deg=38.0,
            power_db=17.0,
            label="Source 3",
        ),
    ],
    noise_power_db=0.0,
    seed=42,
)

model_order = estimate_source_count_mdl(
    covariance_matrix=simulation.sample_covariance,
    number_of_snapshots=number_of_snapshots,
    maximum_sources=6,
)

print(
    "MDL estimated source count:",
    model_order.estimated_number_of_sources,
)

music = calculate_music_spectrum(
    covariance_matrix=simulation.sample_covariance,
    number_of_elements=number_of_elements,
    frequency_hz=frequency_hz,
    element_spacing_m=wavelength_m / 2.0,
    number_of_sources=model_order.estimated_number_of_sources,
)

capon = calculate_capon_spectrum(
    covariance_matrix=simulation.sample_covariance,
    number_of_elements=number_of_elements,
    frequency_hz=frequency_hz,
    element_spacing_m=wavelength_m / 2.0,
    number_of_sources=model_order.estimated_number_of_sources,
)

bartlett = calculate_bartlett_spectrum(
    covariance_matrix=simulation.sample_covariance,
    number_of_elements=number_of_elements,
    frequency_hz=frequency_hz,
    element_spacing_m=wavelength_m / 2.0,
    number_of_sources=model_order.estimated_number_of_sources,
)

errors = calculate_direction_errors(
    detected_angles_deg=music.detected_angles_deg,
    true_angles_deg=true_angles_deg,
)

print(
    "MUSIC detected angles:",
    music.detected_angles_deg,
)

print(
    "MUSIC RMSE:",
    errors["root_mean_square_error_deg"],
)

plt.figure(figsize=(11, 6))

plt.plot(
    bartlett.scan_angles_deg,
    bartlett.spectrum_db,
    label="Bartlett",
)

plt.plot(
    capon.scan_angles_deg,
    capon.spectrum_db,
    label="Capon",
)

plt.plot(
    music.scan_angles_deg,
    music.spectrum_db,
    label="MUSIC",
)

for true_angle in true_angles_deg:
    plt.axvline(
        true_angle,
        linestyle=":",
    )

plt.xlabel("Angle of arrival (degrees)")
plt.ylabel("Normalized spatial spectrum (dB)")
plt.title("Bartlett, Capon and MUSIC Direction Finding")
plt.xlim(-90, 90)
plt.ylim(-60, 0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
