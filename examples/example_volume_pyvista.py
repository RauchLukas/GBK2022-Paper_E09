
import pyvista
import numpy as np
import vtk

def volume_pyvista():
    print("\n=====================================================")
    print("EXAMPLE: Volume Calculus with PyVista")
    print("=====================================================\n")

    p = pyvista.Plotter()
    p.show_bounds()

    # mesh points
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]])

    # VOLUME
    faces = np.hstack([[4, 0, 1, 2, 3], [3, 0, 1, 4], [3, 1, 2, 4], [3, 2, 3, 4], [3, 3, 0, 4]])  # [square, triangle, triangle, triangle, triangle]
    pyramid = pyvista.PolyData(vertices, faces)
    vol = pyramid.volume
    print("Volume pyramid = {:1.6} \tis close to {:1.6}" .format(vol, 1/3))
    assert np.isclose(pyramid.volume, 1/3)

    # SURFACE
    faces = np.hstack([[3, 0, 1, 4], [3, 1, 2, 4], [3, 2, 3, 4], [3, 3, 0, 4]])  # [triangle, triangle, triangle, triangle]
    surf = pyvista.PolyData(vertices, faces)
    surf.triangulate(inplace=True)
    p.add_mesh(surf, style='wireframe', color='r')

    # Cutting Plane
    h = 1
    trim_surf = pyvista.Plane(center=(surf.center[0], surf.center[1], h), direction=(0,0,1), i_size=2, j_size=2)
    p.add_mesh(trim_surf, color='r', opacity=0.3)

    alg = vtk.vtkTrimmedExtrusionFilter()
    alg.SetInputData(0, surf)
    alg.SetInputData(1, trim_surf)
    alg.SetCappingStrategy(0) # <-- ensure that the cap is defined by the intersection
    alg.SetExtrusionDirection(0, 0, h) # <-- set this with the plane normal
    alg.Update()
    # output = pyvista.core.filters._get_output(alg)
    output = pyvista.core.filters._get_output(alg).compute_normals()


    # Checking Results
    volume_goal = abs(pyramid.volume - h)
    print("Volume extrusion = {:1.6} \tis close to {:1.6}" .format(output.volume, volume_goal))
    assert np.isclose(output.volume, volume_goal)

    p.add_mesh(output, color='w', opacity=0.3)

    p.show()