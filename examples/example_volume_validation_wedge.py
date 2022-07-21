# Author: Lukas Rauch
# Data: 18.05.2022

from scripts.mesh import Mesh
from scripts.projection_plane import Plane, Projection

import open3d as o3d
import numpy as np
from numpy import genfromtxt
import pyvista as pv
import vtk
import time


def volume_validation_wedge():
    print("\n============================================================================")
    print("EXAMPLE: synthetic Point Cloud Surface Reconstruction and volume calculus")
    print("============================================================================\n")

    # Set Visualization Toggle
    SHOW_O3D = False
    SHOW_PV = True

    path = ""

    # Import {x,y,z} coordinates from csv
    path = "./samples/6e_pyramid.csv"
    data = genfromtxt(path, delimiter=',')
    print("--> Importing: ", path)

    """
      Object Reconstruction
      
      with the help of Open3D
      https://github.com/isl-org/Open3D
      """

    # Create open3D Point Cloud Database
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    print("===> ", pcd)

    # Create a Coordinate Frame
    xyz = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    # Visualize Point Cloud
    if SHOW_O3D:
        o3d.visualization.draw_geometries([pcd, xyz])

    # Normals
    radius = 0.1
    nn = 30
    print("--> Estimate Normals within radius={} and max. nearest_neighbours={} ...".format(radius, nn))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nn))

    px = 0
    py = 0
    pz = 10000
    print("--> Orient Normals towards position ({}, {}, {})".format(px, py, pz))
    pcd.orient_normals_towards_camera_location(camera_location=([px, py, pz]))
    if SHOW_O3D:
        o3d.visualization.draw_geometries([pcd, xyz], point_show_normal=True)

    # Poisson Meshing
    d = 5
    print("--> Poisson Meshing with Octree Depth d={} ...".format(d))
    t = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=d, width=0, scale=1.1,
                                                                                linear_fit=True, n_threads=-1)
    print("===> Poisson Meshing [DONE], process time = {:10.4f} seconds".format(time.time() - t))

    if SHOW_O3D:
        print("--> Visualize Mesh:")
        print("\tINFO: Mesh Colors can be changes with Crtl + 0..4,9:")
        print("\t\tCrtl + 0 - Default Color "
              "\n\t\tCrtl + 1 - Render Point Color"
              "\n\t\tCrtl + 2 - x coordinate as color"
              "\n\t\tCrtl + 3 - y coordinate as color"
              "\n\t\tCrtl + 4 - z coordinate as color"
              "\n\t\tCrtl + 9 - normals as color")
        o3d.visualization.draw_geometries([mesh, pcd, xyz], mesh_show_wireframe=True, mesh_show_back_face=True)

    print("----------------------------------------------")


    print("--> Convert open3D cloud to pyvista PolyData")
    v = np.asarray(mesh.vertices)
    f = np.array(mesh.triangles)

    pv_faces = np.ones((f.shape[0], 4), dtype=int) * 3
    pv_faces[:, 1:] = f

    mesh = pv.PolyData(v, pv_faces)
    clip_origin = (0, 0, 0)
    clip_normal = (0, 0, 1)
    print("--> Clip mesh at origin: ", clip_origin, " normal: ", clip_normal)
    mesh = mesh.clip(origin=(0, 0, 0), normal=(0, 0, 1), invert=False)
    # print(mesh.plot(show_bounds=True))

    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    my_mesh = Mesh(vertices, faces)

    plane = Plane([0, 0, 1, 0.25])

    proj = Projection(mesh=my_mesh, plane=plane)

    print("--> Project Mesh Points to plane [a, b, c, d] = ", plane )
    projected_verts, _ = proj.project_points(vertices)

    projected_plane = pv.PolyData(projected_verts, mesh.faces)
    projected_verts, _ = proj.project_points(my_mesh.vertices)

    print("--> Calculate Mesh surface area.")
    mesh_area = proj.calculate_triangle_area()
    print("==> Mesh Area SUM = {:.f5}" .format(np.sum(mesh_area)))

    print("--> Calculate projected Volume.")
    volume = proj.volume_mesh_projected(vertices, faces)
    print("==> Volume SUM = {:.f5}" .format(np.sum(volume)))
    mesh['volume'] = volume
    mesh['area'] = mesh_area

    projected_plane['volume'] = volume


    print("--> Plot Results.")
    pv.set_plot_theme("document")
    plotter = pv.Plotter(shape=(1, 2))
    plotter.link_views()  # link all the views

    plotter.subplot(0, 0)
    plotter.add_text("Surface Area, summ: {:.5f}".format(np.sum(mesh_area), font_size=10))
    mesh.set_active_scalars('area')
    plotter.add_mesh(mesh.copy(), show_edges=True)

    plotter.subplot(0, 1)

    mesh.set_active_scalars('volume')
    plotter.add_text("Volume, summ: {:.5f}".format(np.sum(volume), font_size=10))
    plotter.add_mesh(mesh.copy(), show_edges=True, opacity=.3)
    plotter.add_mesh(projected_plane.copy(), show_edges=True)

    cpos = [(-1.0350912900806826, -2.074382590116559, 1.9888771401591017),
            (1.1770011373844451, 1.2261028636872973, -0.31919989645238067),
            (0.25874385336700145, 0.4310820182173556, 0.8644188289911594)]

    cpos = plotter.show(cpos=cpos, full_screen=True, return_cpos=True)
