from scripts.mesh import Mesh
from scripts.projection_plane import Plane, Projection

import numpy as np
from copy import copy
import pyvista as pv
from pyvista import examples


def wedges_terrain():
    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background('black', top='white')
    plotter.link_views()  # link all the views

    cpos = [(-9.333824315352208, -19.711062527056335, 9.560934366224249),
     (2.873191678685852, 0.9646737608936182, 0.8324680150317425),
     (0.16224806279966864, 0.30093000785178387, 0.939742888503063)]

    dem = examples.download_crater_topo()
    subset = dem.extract_subset((500, 900, 400, 800, 0, 0), (20, 20, 1))
    terrain = subset.warp_by_scalar()
    terrain = terrain.triangulate()
    terrain.clear_data()

    print(terrain.cells.shape)


    # terrain.add_field_data(np.random.random(terrain.cells.shape[0]), 'data')
    terrain.cell_data["values"] = np.random.random(800)
    terrain.cell_data["range"] = np.arange(800)


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
    pv_mesh = pv.PolyData(projected_verts, projected_faces)

    terrain.cell_data["area"] = mesh_area
    terrain.set_active_scalars('area')

    # terrain.plot(show_edges=True)

    plotter.subplot(0, 0)
    plotter.add_text("terrain", font_size=10)
    plotter.add_mesh(terrain, scalars="area", show_edges=True, scalar_bar_args={'title': 'Area'})
    # plotter.scalar_bar("zzz")

    plotter.subplot(0, 1)
    plotter.add_text("volume", font_size=10)
    plotter.add_mesh(terrain.copy(), scalars=volume, show_edges=True, scalar_bar_args={'title': "Volume"})
    plotter.add_mesh(pv_mesh.copy(), scalars=volume, show_edges=True, scalar_bar_args={'title': "Volume"})

    # Display the window
    cpos = plotter.show(cpos=cpos, full_screen=True)

    print(cpos)


