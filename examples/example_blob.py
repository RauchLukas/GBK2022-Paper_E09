from scipy import integrate
import pyvista as pv
import numpy as np
from time import time

from scripts.mesh import Mesh
from scripts.projection_plane import Plane, Projection


def example_blob():

    x0, x1 = 0, 1
    y0, y1 = 0, 1

    d = 8

    function_z = lambda x, y: d * (1 - x) * x * (1 - y) * y

    # reference value
    goal = integrate.dblquad(function_z, y0, y1, x0, y1)[0]

    x = np.linspace(0., 1., 21)
    y = np.linspace(0., 1., 21)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    z = function_z(xx, yy)

    # 2D surface mesh
    mesh = pv.RectilinearGrid(x, y).extract_geometry()
    mesh.points[:, 2] = z.ravel()

    mesh.triangulate(inplace=True)

    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    my_mesh = Mesh(vertices, faces)

    projection_plane = [0, 0, 1, 0]
    p_plane = Plane(projection_plane)

    projection = Projection(mesh=my_mesh, plane=p_plane)

    print("--> Calculate Mesh surface area.")
    s = time()
    mesh_area = projection.calculate_triangle_area()
    print("==> Mesh Area SUM = {:.5f}" .format(np.sum(mesh_area)))
    print("\t[INFO]: time: {:.5f} sec.".format(time() - s))

    print("--> Project Mesh Points to plane [a, b, c, d] = ", p_plane.plane)
    s = time()
    projected_verts, _ = projection.project_points(vertices)
    print("\t[INFO]: time: {:.5f} sec." .format(time()-s))

    print("--> Calculate projected Volume.")
    s = time()
    volume = projection.volume_mesh_projected(vertices, faces)
    print("==> Volume SUM = {vol:.5f}, goal was = {goal:.5f}" .format(vol=np.sum(volume), goal=goal))
    eps = goal-np.sum(volume)
    error = 100 * eps / goal
    print("==> ERROR = {eps:.5f}, eps = {error:.3f}%".format(eps=eps, error=error))
    print("\t[INFO]: time: {:.5f} sec.".format(time() - s))

    mesh['volume'] = volume
    mesh['area'] = mesh_area

    projected_plane = pv.PolyData(projected_verts, mesh.faces)
    # projected_verts, _ = projected_plane.project_points(my_mesh.vertices)

    projected_plane['volume'] = volume

    print("--> Plot Results.")
    pv.set_plot_theme("document")
    plotter = pv.Plotter(shape=(1, 2))
    plotter.link_views()  # link all the views

    plotter.subplot(0, 0)
    plotter.add_text(
        "Surface Area, sum = {sum:.5f} [units²]"
        "\n F(x, y) = {fact:.2f} * (1-x) * x * (1-y) * y"
        .format(
            sum=np.sum(mesh_area),
            fact=d)
        , font_size=16)
    mesh.set_active_scalars('area')
    plotter.add_mesh(mesh.copy(), show_edges=True, scalar_bar_args={'title': "Area [units²]"})

    plotter.subplot(0, 1)

    mesh.set_active_scalars('volume')
    plotter.add_text(
        "Volume, sum = {vol:.5f}[units³]"
        "\neps = {eps:.5f}[units³], error = {error:.3f}%"
        .format(
            vol=np.sum(volume),
            eps=eps,
            error=error),
            font_size=16)

    plotter.add_mesh(mesh.copy(), show_edges=True, opacity=.3, scalar_bar_args={'title': "Volume [units³]"})
    plotter.add_mesh(projected_plane.copy(), show_edges=True, scalar_bar_args={'title': "Volume [units³]"})

    cpos = [(-1.0350912900806826, -2.074382590116559, 1.9888771401591017),
            (1.1770011373844451, 1.2261028636872973, -0.31919989645238067),
            (0.25874385336700145, 0.4310820182173556, 0.8644188289911594)]

    cpos = plotter.show(cpos=cpos, full_screen=True, return_cpos=True)









