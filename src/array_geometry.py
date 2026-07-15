"""Antenna-array geometry generation and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ArrayGeometry:
    """Cartesian coordinates and state information for an antenna array."""

    coordinates_m: NDArray[np.float64]
    active_mask: NDArray[np.bool_]
    element_labels: tuple[str, ...]
    geometry_type: str

    @property
    def number_of_elements(self) -> int:
        """Return the total number of physical array elements."""

        return int(self.coordinates_m.shape[0])

    @property
    def number_of_active_elements(self) -> int:
        """Return the number of active array elements."""

        return int(np.count_nonzero(self.active_mask))

    @property
    def active_coordinates_m(self) -> NDArray[np.float64]:
        """Return coordinates of active antenna elements."""

        return self.coordinates_m[self.active_mask]

    @property
    def failed_element_indices(self) -> NDArray[np.int64]:
        """Return indices of inactive or failed elements."""

        return np.flatnonzero(~self.active_mask).astype(np.int64)


def create_uniform_linear_array(
    number_of_elements: int,
    element_spacing_m: float,
    *,
    axis: str = "x",
    centered: bool = True,
    failed_element_indices: list[int] | NDArray[np.int64] | None = None,
) -> ArrayGeometry:
    """
    Create a uniform linear array.

    Parameters
    ----------
    number_of_elements:
        Total number of antenna elements.

    element_spacing_m:
        Centre-to-centre spacing between adjacent elements in metres.

    axis:
        Cartesian axis along which the array is placed. Supported
        values are ``x``, ``y``, and ``z``.

    centered:
        When True, position the array symmetrically around the origin.

    failed_element_indices:
        Optional indices of inactive antenna elements.

    Returns
    -------
    ArrayGeometry
        Linear-array coordinates and active-element mask.
    """

    _validate_number_of_elements(number_of_elements)
    _validate_positive_finite(
        element_spacing_m,
        name="element_spacing_m",
    )

    normalized_axis = axis.strip().lower()

    axis_lookup = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    if normalized_axis not in axis_lookup:
        raise ValueError(
            "axis must be one of: 'x', 'y', or 'z'."
        )

    positions = (
        np.arange(number_of_elements, dtype=np.float64)
        * element_spacing_m
    )

    if centered:
        positions -= np.mean(positions)

    coordinates = np.zeros(
        (number_of_elements, 3),
        dtype=np.float64,
    )

    coordinates[:, axis_lookup[normalized_axis]] = positions

    active_mask = _create_active_mask(
        number_of_elements=number_of_elements,
        failed_element_indices=failed_element_indices,
    )

    labels = tuple(
        f"E{index}"
        for index in range(number_of_elements)
    )

    return ArrayGeometry(
        coordinates_m=coordinates,
        active_mask=active_mask,
        element_labels=labels,
        geometry_type="uniform_linear_array",
    )


def create_rectangular_planar_array(
    number_of_rows: int,
    number_of_columns: int,
    row_spacing_m: float,
    column_spacing_m: float,
    *,
    plane: str = "xy",
    centered: bool = True,
    failed_element_indices: list[int] | NDArray[np.int64] | None = None,
) -> ArrayGeometry:
    """
    Create a rectangular planar antenna array.

    Elements are numbered row-by-row.

    Parameters
    ----------
    number_of_rows:
        Number of element rows.

    number_of_columns:
        Number of element columns.

    row_spacing_m:
        Spacing between adjacent rows in metres.

    column_spacing_m:
        Spacing between adjacent columns in metres.

    plane:
        Cartesian plane containing the array. Supported values are
        ``xy``, ``xz``, and ``yz``.

    centered:
        When True, center the array around the origin.

    failed_element_indices:
        Optional flattened indices of inactive elements.
    """

    _validate_positive_integer(
        number_of_rows,
        name="number_of_rows",
    )
    _validate_positive_integer(
        number_of_columns,
        name="number_of_columns",
    )

    if number_of_rows < 2 and number_of_columns < 2:
        raise ValueError(
            "The planar array must contain at least two elements."
        )

    _validate_positive_finite(
        row_spacing_m,
        name="row_spacing_m",
    )
    _validate_positive_finite(
        column_spacing_m,
        name="column_spacing_m",
    )

    normalized_plane = plane.strip().lower()

    plane_axes = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }

    if normalized_plane not in plane_axes:
        raise ValueError(
            "plane must be one of: 'xy', 'xz', or 'yz'."
        )

    row_positions = (
        np.arange(number_of_rows, dtype=np.float64)
        * row_spacing_m
    )

    column_positions = (
        np.arange(number_of_columns, dtype=np.float64)
        * column_spacing_m
    )

    if centered:
        row_positions -= np.mean(row_positions)
        column_positions -= np.mean(column_positions)

    row_grid, column_grid = np.meshgrid(
        row_positions,
        column_positions,
        indexing="ij",
    )

    total_elements = number_of_rows * number_of_columns

    coordinates = np.zeros(
        (total_elements, 3),
        dtype=np.float64,
    )

    row_axis, column_axis = plane_axes[normalized_plane]

    coordinates[:, row_axis] = row_grid.ravel()
    coordinates[:, column_axis] = column_grid.ravel()

    active_mask = _create_active_mask(
        number_of_elements=total_elements,
        failed_element_indices=failed_element_indices,
    )

    labels = tuple(
        f"R{row}C{column}"
        for row in range(number_of_rows)
        for column in range(number_of_columns)
    )

    return ArrayGeometry(
        coordinates_m=coordinates,
        active_mask=active_mask,
        element_labels=labels,
        geometry_type="rectangular_planar_array",
    )


def create_arbitrary_array(
    coordinates_m: NDArray[np.float64],
    *,
    active_mask: NDArray[np.bool_] | None = None,
    element_labels: list[str] | tuple[str, ...] | None = None,
    geometry_type: str = "arbitrary_array",
) -> ArrayGeometry:
    """
    Create an array from arbitrary three-dimensional coordinates.

    Parameters
    ----------
    coordinates_m:
        Array with shape ``(N, 3)`` containing x, y, and z coordinates.

    active_mask:
        Optional Boolean vector identifying active elements.

    element_labels:
        Optional element names.

    geometry_type:
        Descriptive geometry identifier.
    """

    coordinates = np.asarray(
        coordinates_m,
        dtype=np.float64,
    )

    _validate_coordinates(coordinates)

    number_of_elements = coordinates.shape[0]

    if active_mask is None:
        prepared_active_mask = np.ones(
            number_of_elements,
            dtype=bool,
        )
    else:
        prepared_active_mask = np.asarray(
            active_mask,
            dtype=bool,
        )

        if prepared_active_mask.shape != (number_of_elements,):
            raise ValueError(
                "active_mask must have shape "
                f"({number_of_elements},)."
            )

        if not np.any(prepared_active_mask):
            raise ValueError(
                "At least one array element must remain active."
            )

    if element_labels is None:
        prepared_labels = tuple(
            f"E{index}"
            for index in range(number_of_elements)
        )
    else:
        prepared_labels = tuple(
            str(label)
            for label in element_labels
        )

        if len(prepared_labels) != number_of_elements:
            raise ValueError(
                "element_labels must contain one label per element."
            )

        if len(set(prepared_labels)) != len(prepared_labels):
            raise ValueError(
                "element_labels must be unique."
            )

    cleaned_geometry_type = str(geometry_type).strip()

    if not cleaned_geometry_type:
        raise ValueError(
            "geometry_type cannot be empty."
        )

    return ArrayGeometry(
        coordinates_m=coordinates.copy(),
        active_mask=prepared_active_mask.copy(),
        element_labels=prepared_labels,
        geometry_type=cleaned_geometry_type,
    )


def apply_position_errors(
    geometry: ArrayGeometry,
    standard_deviation_m: float,
    *,
    seed: int | None = None,
    axes: str = "xyz",
) -> ArrayGeometry:
    """
    Apply independent Gaussian position errors to array elements.

    Parameters
    ----------
    geometry:
        Original antenna-array geometry.

    standard_deviation_m:
        Standard deviation of element-position errors in metres.

    seed:
        Optional random seed.

    axes:
        Axes along which errors are applied. Examples: ``x``, ``xy``,
        or ``xyz``.
    """

    _validate_geometry(geometry)

    if not np.isfinite(standard_deviation_m):
        raise ValueError(
            "standard_deviation_m must be finite."
        )

    if standard_deviation_m < 0.0:
        raise ValueError(
            "standard_deviation_m cannot be negative."
        )

    normalized_axes = axes.strip().lower()

    if not normalized_axes:
        raise ValueError("axes cannot be empty.")

    if any(axis not in "xyz" for axis in normalized_axes):
        raise ValueError(
            "axes may contain only 'x', 'y', and 'z'."
        )

    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError(
            "axes cannot contain repeated axis names."
        )

    perturbed_coordinates = geometry.coordinates_m.copy()

    if standard_deviation_m > 0.0:
        random_generator = np.random.default_rng(seed)

        axis_lookup = {
            "x": 0,
            "y": 1,
            "z": 2,
        }

        for axis in normalized_axes:
            perturbed_coordinates[:, axis_lookup[axis]] += (
                random_generator.normal(
                    loc=0.0,
                    scale=standard_deviation_m,
                    size=geometry.number_of_elements,
                )
            )

    return ArrayGeometry(
        coordinates_m=perturbed_coordinates,
        active_mask=geometry.active_mask.copy(),
        element_labels=geometry.element_labels,
        geometry_type=f"{geometry.geometry_type}_with_position_errors",
    )


def set_failed_elements(
    geometry: ArrayGeometry,
    failed_element_indices: list[int] | NDArray[np.int64],
) -> ArrayGeometry:
    """Return a new geometry with selected elements marked inactive."""

    _validate_geometry(geometry)

    active_mask = _create_active_mask(
        number_of_elements=geometry.number_of_elements,
        failed_element_indices=failed_element_indices,
    )

    return ArrayGeometry(
        coordinates_m=geometry.coordinates_m.copy(),
        active_mask=active_mask,
        element_labels=geometry.element_labels,
        geometry_type=geometry.geometry_type,
    )


def reactivate_all_elements(
    geometry: ArrayGeometry,
) -> ArrayGeometry:
    """Return a copy of the geometry with every element active."""

    _validate_geometry(geometry)

    return ArrayGeometry(
        coordinates_m=geometry.coordinates_m.copy(),
        active_mask=np.ones(
            geometry.number_of_elements,
            dtype=bool,
        ),
        element_labels=geometry.element_labels,
        geometry_type=geometry.geometry_type,
    )


def calculate_pairwise_distances(
    geometry: ArrayGeometry,
    *,
    active_only: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate the Euclidean distance between every pair of elements.

    Returns an ``N × N`` symmetric distance matrix.
    """

    _validate_geometry(geometry)

    if active_only:
        coordinates = geometry.active_coordinates_m
    else:
        coordinates = geometry.coordinates_m

    coordinate_differences = (
        coordinates[:, np.newaxis, :]
        - coordinates[np.newaxis, :, :]
    )

    return np.linalg.norm(
        coordinate_differences,
        axis=2,
    ).astype(np.float64)


def calculate_array_extent_m(
    geometry: ArrayGeometry,
) -> NDArray[np.float64]:
    """
    Return the physical array extent along x, y, and z.

    The result has the form ``[x_extent, y_extent, z_extent]``.
    """

    _validate_geometry(geometry)

    minimum_coordinates = np.min(
        geometry.coordinates_m,
        axis=0,
    )

    maximum_coordinates = np.max(
        geometry.coordinates_m,
        axis=0,
    )

    return (
        maximum_coordinates - minimum_coordinates
    ).astype(np.float64)


def _create_active_mask(
    number_of_elements: int,
    failed_element_indices: list[int] | NDArray[np.int64] | None,
) -> NDArray[np.bool_]:
    """Create an active-element mask from failed-element indices."""

    active_mask = np.ones(
        number_of_elements,
        dtype=bool,
    )

    if failed_element_indices is None:
        return active_mask

    failed_indices = np.asarray(
        failed_element_indices,
        dtype=np.int64,
    )

    if failed_indices.ndim != 1:
        raise ValueError(
            "failed_element_indices must be one-dimensional."
        )

    if failed_indices.size == 0:
        return active_mask

    if np.any(failed_indices < 0) or np.any(
        failed_indices >= number_of_elements
    ):
        raise IndexError(
            "failed_element_indices contains an index outside "
            "the valid element range."
        )

    if np.unique(failed_indices).size != failed_indices.size:
        raise ValueError(
            "failed_element_indices contains duplicate indices."
        )

    active_mask[failed_indices] = False

    if not np.any(active_mask):
        raise ValueError(
            "At least one antenna element must remain active."
        )

    return active_mask


def _validate_coordinates(
    coordinates_m: NDArray[np.float64],
) -> None:
    """Validate an antenna-coordinate matrix."""

    if coordinates_m.ndim != 2:
        raise ValueError(
            "coordinates_m must be a two-dimensional array."
        )

    if coordinates_m.shape[1] != 3:
        raise ValueError(
            "coordinates_m must have shape (N, 3)."
        )

    if coordinates_m.shape[0] < 2:
        raise ValueError(
            "At least two antenna elements are required."
        )

    if not np.all(np.isfinite(coordinates_m)):
        raise ValueError(
            "coordinates_m contains non-finite values."
        )

    unique_coordinates = np.unique(
        coordinates_m,
        axis=0,
    )

    if unique_coordinates.shape[0] != coordinates_m.shape[0]:
        raise ValueError(
            "Two or more antenna elements occupy the same coordinate."
        )


def _validate_geometry(
    geometry: ArrayGeometry,
) -> None:
    """Validate an ArrayGeometry object."""

    if not isinstance(geometry, ArrayGeometry):
        raise TypeError(
            "geometry must be an ArrayGeometry object."
        )

    _validate_coordinates(geometry.coordinates_m)

    if geometry.active_mask.shape != (
        geometry.number_of_elements,
    ):
        raise ValueError(
            "geometry.active_mask has an invalid shape."
        )

    if len(geometry.element_labels) != geometry.number_of_elements:
        raise ValueError(
            "geometry.element_labels has an invalid length."
        )

    if not np.any(geometry.active_mask):
        raise ValueError(
            "At least one antenna element must remain active."
        )


def _validate_number_of_elements(
    number_of_elements: int,
) -> None:
    """Validate the number of antenna elements."""

    _validate_positive_integer(
        number_of_elements,
        name="number_of_elements",
    )

    if number_of_elements < 2:
        raise ValueError(
            "number_of_elements must be at least 2."
        )


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> None:
    """Validate a positive integer."""

    if isinstance(value, bool) or not isinstance(
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