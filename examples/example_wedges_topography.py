from scripts.mesh import Mesh
from scripts.projection_plane import Plane, Projection

import numpy as np
import pyvista as pv
from pyvista import examples


def wedges_terrain():
    plotter = pv.Plotter(shape=(1, 2))
    plotter.link_views()  # link all the views

    dem = examples.download_crater_topo()
    subset = dem.extract_subset((500, 900, 400, 800, 0, 0), (20, 20, 1))
    terrain = subset.warp_by_scalar()
    terrain = terrain.triangulate()

    # terrain.plot(show_edges=True)



    v = terrain.points
    f = terrain.cells.reshape(-1, 4)[:, 1:]

    mini = np.min(v, axis=0)

    v[:, 0] -= mini[0]
    v[:, 1] -= mini[1]
    v[:, 2] -= mini[2]

    maxi = np.max(v, axis=0)

    v[:, 0] /= maxi[0] / 10
    v[:, 1] /= maxi[1] / 10
    v[:, 2] /= maxi[2] / 5

    mesh = Mesh(v, f)

    plane = Plane([0, 0, 1, 2])
    c_xyz = plane.p_origin

    proj = Projection(mesh=mesh, plane=plane)

    projected_verts = terrain.points
    projected_faces = terrain.cells

    print(projected_verts.shape)
    projected_verts, _ = proj.project_points(mesh.vertices)

    volume = proj.volume_mesh_projected(v, f)
    print("volumes: ", np.sum(volume))

    mesh_area = proj.calculate_triangle_area()
    print(np.max(mesh_area))
    print(np.min(mesh_area))
    pv_mesh = pv.PolyData(projected_verts, projected_faces)

    plotter.subplot(0, 0)
    plotter.add_text("terrain", font_size=10)
    plotter.add_mesh(terrain,
                     scalars=mesh_area, show_edges=True, scalar_bar_args={'title': "Area"})
    # plotter.scalar_bar("zzz")

    plotter.subplot(0, 1)
    plotter.add_text("volume", font_size=10)
    plotter.add_mesh(terrain, scalars=volume, show_edges=True, scalar_bar_args={'title': "Volume"})
    plotter.add_mesh(pv_mesh, scalars=volume, show_edges=True, scalar_bar_args={'title': "Volume"})


    # Display the window
    plotter.show()


