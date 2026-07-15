"""General three-dimensional antenna-array beamforming calculations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from src.array_geometry import ArrayGeometry
SPEED_OF_LIGHT_M_S = 299_792_458.0
@dataclass(frozen=True)
class GeneralBeamformerResult:
    """Results from an arbitrary-array beamforming calculation."""
    azimuth_deg: NDArray[np.float64]
    elevation_deg: NDArray[np.float64]
    array_factor: NDArray[np.complex128]
    magnitude: NDArray[np.float64]
    magnitude_db: NDArray[np.float64]
    wavelength_m: float
    wave_number_rad_m: float
    active_element_count: int
def direction_unit_vector(
    azimuth_deg: NDArray[np.float64] | float,
    elevation_deg: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """
    Convert azimuth and elevation angles into Cartesian unit vectors.
    Coordinate convention
    ---------------------
    Azimuth is measured in the x-y plane:
    - 0° points along +x
    - 90° points along +y
    - -90° points along -y
    - 180° points along -x
    Elevation is measured above the x-y plane:
    - 0° lies in the x-y plane
    - 90° points along +z
    - -90° points along -z
    Returns
    -------
    numpy.ndarray
        Array with final dimension 3 containing x, y, and z direction
        components.
    """
    azimuth = np.asarray(
        azimuth_deg,
        dtype=np.float64,
    )
    elevation = np.asarray(
        elevation_deg,
        dtype=np.float64,
    )
    try:
        azimuth, elevation = np.broadcast_arrays(
            azimuth,
            elevation,
        )
    except ValueError as error:
        raise ValueError(
            "azimuth_deg and elevation_deg must be broadcast-compatible."
        ) from error
    if not np.all(np.isfinite(azimuth)):
        raise ValueError(
            "azimuth_deg contains non-finite values."
        )
    if not np.all(np.isfinite(elevation)):
        raise ValueError(
            "elevation_deg contains non-finite values."
        )
    if np.any(elevation < -90.0) or np.any(
        elevation > 90.0
    ):
        raise ValueError(
            "elevation_deg must lie between -90 and 90 degrees."
        )
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)
    cosine_elevation = np.cos(elevation_rad)
    x_component = (
        cosine_elevation * np.cos(azimuth_rad)
    )
    y_component = (
        cosine_elevation * np.sin(azimuth_rad)
    )
    z_component = np.sin(elevation_rad)
    return np.stack(
        (
            x_component,
            y_component,
            z_component,
        ),
        axis=-1,
    ).astype(np.float64)
def generate_geometric_steering_weights(
    geometry: ArrayGeometry,
    frequency_hz: float,
    steering_azimuth_deg: float,
    steering_elevation_deg: float,
    *,
    amplitude_weights: NDArray[np.float64] | None = None,
    include_inactive_elements: bool = True,
    normalize_amplitude: bool = True,
) -> NDArray[np.complex128]:
    """
    Generate complex steering weights from arbitrary 3D coordinates.
    The excitation applied to element n is
        w_n = a_n exp(-j k r_n · u_0)
    where:
    - ``a_n`` is the amplitude weight,
    - ``r_n`` is the element-position vector,
    - ``u_0`` is the requested steering-direction unit vector.
    """
    _validate_geometry(geometry)
    _validate_frequency(frequency_hz)
    steering_direction = direction_unit_vector(
        steering_azimuth_deg,
        steering_elevation_deg,
    )
    wave_number_rad_m = (
        2.0
        * np.pi
        * frequency_hz
        / SPEED_OF_LIGHT_M_S
    )
    amplitudes = _prepare_amplitudes(
        geometry=geometry,
        amplitude_weights=amplitude_weights,
        normalize_amplitude=normalize_amplitude,
    )
    geometric_phase_rad = (
        -wave_number_rad_m
        * (
            geometry.coordinates_m
            @ steering_direction
        )
    )
    complex_weights = (
        amplitudes
        * np.exp(1j * geometric_phase_rad)
    ).astype(np.complex128)
    if include_inactive_elements:
        complex_weights = complex_weights.copy()
        complex_weights[~geometry.active_mask] = 0.0 + 0.0j
        return complex_weights
    return complex_weights[geometry.active_mask]
def calculate_array_factor(
    geometry: ArrayGeometry,
    frequency_hz: float,
    azimuth_deg: NDArray[np.float64],
    elevation_deg: NDArray[np.float64],
    *,
    steering_azimuth_deg: float = 0.0,
    steering_elevation_deg: float = 0.0,
    weights: NDArray[np.complex128] | None = None,
    weights_include_steering: bool = False,
    minimum_db: float = -120.0,
) -> GeneralBeamformerResult:
    """
    Calculate the array factor for arbitrary three-dimensional geometry.
    Parameters
    ----------
    geometry:
        Antenna element coordinates and active-element mask.
    frequency_hz:
        Operating frequency in hertz.
    azimuth_deg:
        Observation azimuth values. The array may be one-dimensional or
        multidimensional.
    elevation_deg:
        Observation elevation values. Must be broadcast-compatible with
        ``azimuth_deg``.
    steering_azimuth_deg:
        Requested beam-steering azimuth. Used only when
        ``weights_include_steering`` is False.
    steering_elevation_deg:
        Requested beam-steering elevation. Used only when
        ``weights_include_steering`` is False.
    weights:
        Optional real or complex element weights. The vector may contain
        either one value per physical element or one value per active
        element.
    weights_include_steering:
        Set to True when supplied complex weights already contain the
        geometric steering phase.
    minimum_db:
        Numerical floor for the normalized dB pattern.
    Returns
    -------
    GeneralBeamformerResult
        Complex and normalized array-factor results.
    Notes
    -----
    When steering is handled internally:
        AF(u) =
            sum_n w_n exp[
                j k r_n · (u - u_0)
            ]
    When complete steering weights are supplied:
        AF(u) =
            sum_n w_n exp[
                j k r_n · u
            ]
    """
    _validate_geometry(geometry)
    _validate_frequency(frequency_hz)
    _validate_minimum_db(minimum_db)
    observation_directions = direction_unit_vector(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )
    azimuth_array, elevation_array = np.broadcast_arrays(
        np.asarray(azimuth_deg, dtype=np.float64),
        np.asarray(elevation_deg, dtype=np.float64),
    )
    original_shape = azimuth_array.shape
    flattened_directions = observation_directions.reshape(
        -1,
        3,
    )
    active_coordinates = geometry.active_coordinates_m
    active_weights = _prepare_complex_weights(
        geometry=geometry,
        weights=weights,
    )
    wavelength_m = SPEED_OF_LIGHT_M_S / frequency_hz
    wave_number_rad_m = 2.0 * np.pi / wavelength_m
    observation_phase = (
        wave_number_rad_m
        * (
            active_coordinates
            @ flattened_directions.T
        )
    )
    if weights_include_steering:
        total_phase = observation_phase
    else:
        steering_direction = direction_unit_vector(
            steering_azimuth_deg,
            steering_elevation_deg,
        )
        steering_phase = (
            wave_number_rad_m
            * (
                active_coordinates
                @ steering_direction
            )
        )
        total_phase = (
            observation_phase
            - steering_phase[:, np.newaxis]
        )
    response_matrix = np.exp(
        1j * total_phase
    )
    array_factor_flat = (
        active_weights @ response_matrix
    )
    array_factor = array_factor_flat.reshape(
        original_shape
    )
    magnitude = np.abs(array_factor).astype(
        np.float64
    )
    maximum_magnitude = float(np.max(magnitude))
    if not np.isfinite(maximum_magnitude):
        raise FloatingPointError(
            "The calculated array factor contains non-finite values."
        )
    if np.isclose(maximum_magnitude, 0.0):
        normalized_magnitude = np.zeros_like(
            magnitude,
            dtype=np.float64,
        )
    else:
        normalized_magnitude = (
            magnitude / maximum_magnitude
        )
    minimum_linear = 10.0 ** (
        minimum_db / 20.0
    )
    magnitude_db = (
        20.0
        * np.log10(
            np.maximum(
                normalized_magnitude,
                minimum_linear,
            )
        )
    )
    magnitude_db = np.maximum(
        magnitude_db,
        minimum_db,
    )
    return GeneralBeamformerResult(
        azimuth_deg=azimuth_array.astype(
            np.float64
        ),
        elevation_deg=elevation_array.astype(
            np.float64
        ),
        array_factor=np.asarray(
            array_factor,
            dtype=np.complex128,
        ),
        magnitude=np.asarray(
            normalized_magnitude,
            dtype=np.float64,
        ),
        magnitude_db=np.asarray(
            magnitude_db,
            dtype=np.float64,
        ),
        wavelength_m=float(wavelength_m),
        wave_number_rad_m=float(
            wave_number_rad_m
        ),
        active_element_count=(
            geometry.number_of_active_elements
        ),
    )
def calculate_azimuth_cut(
    geometry: ArrayGeometry,
    frequency_hz: float,
    *,
    elevation_deg: float = 0.0,
    steering_azimuth_deg: float = 0.0,
    steering_elevation_deg: float = 0.0,
    azimuth_start_deg: float = -180.0,
    azimuth_stop_deg: float = 180.0,
    number_of_points: int = 3601,
    weights: NDArray[np.complex128] | None = None,
    weights_include_steering: bool = False,
    minimum_db: float = -120.0,
) -> GeneralBeamformerResult:
    """Calculate a constant-elevation azimuth pattern cut."""
    _validate_number_of_points(number_of_points)
    if azimuth_stop_deg <= azimuth_start_deg:
        raise ValueError(
            "azimuth_stop_deg must exceed azimuth_start_deg."
        )
    azimuth_values = np.linspace(
        azimuth_start_deg,
        azimuth_stop_deg,
        number_of_points,
        dtype=np.float64,
    )
    elevation_values = np.full_like(
        azimuth_values,
        elevation_deg,
        dtype=np.float64,
    )
    return calculate_array_factor(
        geometry=geometry,
        frequency_hz=frequency_hz,
        azimuth_deg=azimuth_values,
        elevation_deg=elevation_values,
        steering_azimuth_deg=steering_azimuth_deg,
        steering_elevation_deg=steering_elevation_deg,
        weights=weights,
        weights_include_steering=weights_include_steering,
        minimum_db=minimum_db,
    )
def calculate_elevation_cut(
    geometry: ArrayGeometry,
    frequency_hz: float,
    *,
    azimuth_deg: float = 0.0,
    steering_azimuth_deg: float = 0.0,
    steering_elevation_deg: float = 0.0,
    elevation_start_deg: float = -90.0,
    elevation_stop_deg: float = 90.0,
    number_of_points: int = 1801,
    weights: NDArray[np.complex128] | None = None,
    weights_include_steering: bool = False,
    minimum_db: float = -120.0,
) -> GeneralBeamformerResult:
    """Calculate a constant-azimuth elevation pattern cut."""
    _validate_number_of_points(number_of_points)
    if elevation_stop_deg <= elevation_start_deg:
        raise ValueError(
            "elevation_stop_deg must exceed elevation_start_deg."
        )
    if elevation_start_deg < -90.0 or elevation_stop_deg > 90.0:
        raise ValueError(
            "Elevation cut limits must lie between -90 and 90 degrees."
        )
    elevation_values = np.linspace(
        elevation_start_deg,
        elevation_stop_deg,
        number_of_points,
        dtype=np.float64,
    )
    azimuth_values = np.full_like(
        elevation_values,
        azimuth_deg,
        dtype=np.float64,
    )
    return calculate_array_factor(
        geometry=geometry,
        frequency_hz=frequency_hz,
        azimuth_deg=azimuth_values,
        elevation_deg=elevation_values,
        steering_azimuth_deg=steering_azimuth_deg,
        steering_elevation_deg=steering_elevation_deg,
        weights=weights,
        weights_include_steering=weights_include_steering,
        minimum_db=minimum_db,
    )
def calculate_azimuth_elevation_grid(
    geometry: ArrayGeometry,
    frequency_hz: float,
    *,
    steering_azimuth_deg: float = 0.0,
    steering_elevation_deg: float = 0.0,
    azimuth_start_deg: float = -180.0,
    azimuth_stop_deg: float = 180.0,
    elevation_start_deg: float = -90.0,
    elevation_stop_deg: float = 90.0,
    number_of_azimuth_points: int = 361,
    number_of_elevation_points: int = 181,
    weights: NDArray[np.complex128] | None = None,
    weights_include_steering: bool = False,
    minimum_db: float = -120.0,
) -> GeneralBeamformerResult:
    """Calculate a two-dimensional azimuth-elevation pattern grid."""
    _validate_number_of_points(
        number_of_azimuth_points
    )
    _validate_number_of_points(
        number_of_elevation_points
    )
    if azimuth_stop_deg <= azimuth_start_deg:
        raise ValueError(
            "azimuth_stop_deg must exceed azimuth_start_deg."
        )
    if elevation_stop_deg <= elevation_start_deg:
        raise ValueError(
            "elevation_stop_deg must exceed elevation_start_deg."
        )
    if elevation_start_deg < -90.0 or elevation_stop_deg > 90.0:
        raise ValueError(
            "Elevation-grid limits must lie between -90 and 90 degrees."
        )
    azimuth_values = np.linspace(
        azimuth_start_deg,
        azimuth_stop_deg,
        number_of_azimuth_points,
        dtype=np.float64,
    )
    elevation_values = np.linspace(
        elevation_start_deg,
        elevation_stop_deg,
        number_of_elevation_points,
        dtype=np.float64,
    )
    azimuth_grid, elevation_grid = np.meshgrid(
        azimuth_values,
        elevation_values,
        indexing="xy",
    )
    return calculate_array_factor(
        geometry=geometry,
        frequency_hz=frequency_hz,
        azimuth_deg=azimuth_grid,
        elevation_deg=elevation_grid,
        steering_azimuth_deg=steering_azimuth_deg,
        steering_elevation_deg=steering_elevation_deg,
        weights=weights,
        weights_include_steering=weights_include_steering,
        minimum_db=minimum_db,
    )
def locate_pattern_peak(
    result: GeneralBeamformerResult,
) -> tuple[float, float, float]:
    """
    Locate the strongest sampled beam direction.
    Returns
    -------
    tuple
        Peak azimuth, peak elevation, and normalized peak level in dB.
    """
    if result.magnitude_db.size == 0:
        raise ValueError(
            "The beamformer result contains no pattern samples."
        )
    flat_index = int(
        np.argmax(result.magnitude_db)
    )
    peak_index = np.unravel_index(
        flat_index,
        result.magnitude_db.shape,
    )
    return (
        float(result.azimuth_deg[peak_index]),
        float(result.elevation_deg[peak_index]),
        float(result.magnitude_db[peak_index]),
    )
def _prepare_amplitudes(
    geometry: ArrayGeometry,
    amplitude_weights: NDArray[np.float64] | None,
    normalize_amplitude: bool,
) -> NDArray[np.float64]:
    """Prepare real-valued amplitudes for all physical elements."""
    if amplitude_weights is None:
        amplitudes = np.ones(
            geometry.number_of_elements,
            dtype=np.float64,
        )
    else:
        amplitudes = np.asarray(
            amplitude_weights,
            dtype=np.float64,
        )
    if amplitudes.shape != (
        geometry.number_of_elements,
    ):
        raise ValueError(
            "amplitude_weights must contain one value per "
            "physical array element."
        )
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError(
            "amplitude_weights contains non-finite values."
        )
    if np.any(amplitudes < 0.0):
        raise ValueError(
            "amplitude_weights cannot contain negative values."
        )
    if np.allclose(amplitudes, 0.0):
        raise ValueError(
            "amplitude_weights cannot contain only zeros."
        )
    if normalize_amplitude:
        amplitudes = amplitudes / np.max(
            amplitudes
        )
    return amplitudes
def _prepare_complex_weights(
    geometry: ArrayGeometry,
    weights: NDArray[np.complex128] | None,
) -> NDArray[np.complex128]:
    """Prepare weights for active array elements."""
    if weights is None:
        return np.ones(
            geometry.number_of_active_elements,
            dtype=np.complex128,
        )
    prepared_weights = np.asarray(
        weights,
        dtype=np.complex128,
    )
    if prepared_weights.ndim != 1:
        raise ValueError(
            "weights must be one-dimensional."
        )
    total_shape = (
        geometry.number_of_elements,
    )
    active_shape = (
        geometry.number_of_active_elements,
    )
    if prepared_weights.shape == total_shape:
        prepared_weights = prepared_weights[
            geometry.active_mask
        ]
    elif prepared_weights.shape != active_shape:
        raise ValueError(
            "weights must contain either one value per physical "
            "element or one value per active element."
        )
    if not np.all(
        np.isfinite(prepared_weights.real)
    ) or not np.all(
        np.isfinite(prepared_weights.imag)
    ):
        raise ValueError(
            "weights contains non-finite values."
        )
    if np.allclose(
        np.abs(prepared_weights),
        0.0,
    ):
        raise ValueError(
            "Active-element weights cannot contain only zeros."
        )
    return prepared_weights
def _validate_geometry(
    geometry: ArrayGeometry,
) -> None:
    """Validate the supplied antenna-array geometry."""
    if not isinstance(geometry, ArrayGeometry):
        raise TypeError(
            "geometry must be an ArrayGeometry object."
        )
    if geometry.coordinates_m.ndim != 2:
        raise ValueError(
            "geometry.coordinates_m must be two-dimensional."
        )
    if geometry.coordinates_m.shape[1] != 3:
        raise ValueError(
            "geometry.coordinates_m must have shape (N, 3)."
        )
    if not np.all(
        np.isfinite(geometry.coordinates_m)
    ):
        raise ValueError(
            "geometry.coordinates_m contains non-finite values."
        )
    if geometry.active_mask.shape != (
        geometry.number_of_elements,
    ):
        raise ValueError(
            "geometry.active_mask has an invalid shape."
        )
    if not np.any(geometry.active_mask):
        raise ValueError(
            "At least one array element must remain active."
        )
def _validate_frequency(
    frequency_hz: float,
) -> None:
    """Validate an operating frequency."""
    if not np.isfinite(frequency_hz):
        raise ValueError(
            "frequency_hz must be finite."
        )
    if frequency_hz <= 0.0:
        raise ValueError(
            "frequency_hz must be positive."
        )
def _validate_minimum_db(
    minimum_db: float,
) -> None:
    """Validate the dB numerical floor."""
    if not np.isfinite(minimum_db):
        raise ValueError(
            "minimum_db must be finite."
        )
    if minimum_db >= 0.0:
        raise ValueError(
            "minimum_db must be below 0 dB."
        )
def _validate_number_of_points(
    number_of_points: int,
) -> None:
    """Validate an angular sample count."""
    if isinstance(number_of_points, bool) or not isinstance(
        number_of_points,
        (int, np.integer),
    ):
        raise TypeError(
            "number_of_points must be an integer."
        )
    if number_of_points < 2:
        raise ValueError(
            "number_of_points must be at least 2."
        )