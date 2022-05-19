# Method from PyVista
# https://github.com/pyvista/pyvista/blob/ebda32c6caad666890efef18bf6ea1f4b9550e23/pyvista/core/filters/poly_data.py#L2


import collections.abc
import numpy as np

import pyvista
from pyvista import (
    NORMALS,
    _vtk,
    abstract_class,
    assert_empty_kwargs,
    generate_plane,
    get_array_association,
    vtk_id_list_to_array,
)
from pyvista.core.errors import DeprecationError, NotAllTrianglesError, VTKVersionError
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.utilities.misc import PyvistaFutureWarning


def extrude_trim(
        poly_data,
        direction,
        trim_surface,
        extrusion="boundary_edges",
        capping="intersection",
        inplace=False,
        progress_bar=False,
):
    """Extrude polygonal data trimmed by a surface.
    This generates polygonal data on output.
    The input dataset is swept along a specified direction forming a
    "skirt" from the boundary edges 2D primitives (i.e., edges used
    by only one polygon); and/or from vertices and lines. The extent
    of the sweeping is defined where the sweep intersects a
    user-specified surface.
    Parameters
    ----------
    direction : numpy.ndarray or sequence
        Direction vector to extrude.
    trim_surface : pyvista.PolyData
        Surface which trims the surface.
    extrusion : str or int, optional
        Control the strategy of extrusion. One of the following:
        * ``"boundary_edges"`` or ``0``
        * ``"all_edges"`` or ``1``
        The default is "boundary_edges".
    capping : str or int, optional
        Control the strategy of capping. One of the following:
        * ``"intersection"`` or ``0``
        * ``"minimum_distance"`` or ``1``
        * ``"maximum_distance"`` or ``2``
        * ``"average_distance"`` or ``3``
        The default is "intersection".
    inplace : bool, optional
        Overwrites the original mesh in-place.
    progress_bar : bool, optional
        Display a progress bar to indicate progress.
    Returns
    -------
    pyvista.PolyData
    Examples
    --------
    Extrude a beam from surface.
    # >>> import pyvista
    # >>> import numpy as np
    # >>> direction = np.array([0, 0, 2])
    # >>> plane = pyvista.Plane()
    # >>> disc = pyvista.Disc(center=(0, 0, 0))
    # >>> extruded = disc.extrude_trim(direction, plane)
    # >>> extruded.plot(color="tan")
    """
    if not isinstance(direction, (np.ndarray, collections.abc.Sequence)) or len(direction) != 3:
        raise TypeError('Vector must be a length three vector')

    extrusions = {"boundary_edges": 0, "all_edges": 1}
    if isinstance(extrusion, str):
        if extrusion not in extrusions:
            raise ValueError(f'Invalid strategy of extrusion "{extrusion}".')
        extrusion = extrusions[extrusion]
    elif isinstance(extrusion, int) and extrusion not in range(2):
        raise ValueError(f'Invalid strategy of extrusion index "{extrusion}".')
    elif not isinstance(extrusion, int):
        raise ValueError(
            f'Invalid strategy of extrusion index type "{type(extrusion).__name__}".'
        )

    cappings = {
        "intersection": 0,
        "minimum_distance": 1,
        "maximum_distance": 2,
        "average_distance": 3,
    }
    if isinstance(capping, str):
        if capping not in cappings:
            raise ValueError(f'Invalid strategy of capping "{capping}".')
        capping = cappings[capping]
    elif isinstance(capping, int) and capping not in range(4):
        raise ValueError(f'Invalid strategy of capping index "{capping}".')

    alg = _vtk.vtkTrimmedExtrusionFilter()
    alg.SetInputData(poly_data)
    alg.SetExtrusionDirection(*direction)
    alg.SetTrimSurfaceData(trim_surface)
    alg.SetExtrusionStrategy(extrusion)
    alg.SetCappingStrategy(capping)
    _update_alg(alg, progress_bar, 'Extruding with trimming')
    output = pyvista.wrap(alg.GetOutput())
    if inplace:
        poly_data.overwrite(output)
        return poly_data
    return output




### Test

direction = (0, 0, -1)

mesh = pyvista.Plane(center=(0, 0, 0), direction=direction, i_size=1, j_size=1, i_resolution=10, j_resolution=10)
plane = pyvista.Plane(center=(0, 0, -1), direction=direction, i_size=2, j_size=2, i_resolution=10, j_resolution=10)

extrusion = extrude_trim(
    poly_data=mesh,
    direction=direction,
    trim_surface=plane,
    )


print(extrusion.volume)

