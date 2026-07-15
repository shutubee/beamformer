import matplotlib.pyplot as plt

from src.beamformer import calculate_ula_array_factor


result = calculate_ula_array_factor(
    number_of_elements=16,
    frequency_hz=28e9,
    element_spacing_m=0.005,
    steering_angle_deg=25.0,
)

plt.figure(figsize=(10, 5))
plt.plot(result.angles_deg, result.magnitude_db)

plt.xlabel("Observation angle (degrees)")
plt.ylabel("Normalized array factor (dB)")
plt.title("16-Element Uniform Linear Array Steered to 25°")
plt.ylim(-60, 0)
plt.xlim(-90, 90)
plt.grid(True)

plt.show()