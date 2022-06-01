# Author: Lukas Rauch
# Data: 23.05.2022

import open3d as o3d
import numpy as np
import pyvista as pv
import vtk
import time


def wall_scan():
    print("\n=====================================================")
    print("EXAMPLE: Real World Wall Scan")
    print("=====================================================\n")

    SHOW_O3D = True
    SHOW_PV = True

    path = "../samples/wall.ply"

    pcd = o3d.io.read_point_cloud(path)

    if not pcd.has_points():
        print("Empty Point Cloud. Abort!")
        exit()
    else:
        print(pcd)

    fac = 5
    points = np.asarray(pcd.points)
    points[:, 0] = points[:, 0] * fac

    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate Normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=([1000.0, 0.0, 0.0]))

    # Plane Fitting
    o3d_plane, inliers = pcd.segment_plane(distance_threshold=0.001, ransac_n =3, num_iterations=100)
    [a, b, c, d] = o3d_plane
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Plane Normal Vector of the fitet Plane
    n_plane = - np.array([o3d_plane[0], o3d_plane[1], o3d_plane[2]])

    # Plane
    x = lambda y, z : (-b * y -c * z - d) / a


    # Poisson Meshing
    d = 9
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=d, width=0, scale=1.1, linear_fit=True)

    mesh.compute_triangle_normals()

    # Crop mesh
    bbox = pcd.get_axis_aligned_bounding_box()
    print("Crop at Bounds: ", bbox)
    mesh_crop = mesh.crop(bbox)

    # Plot
    if SHOW_O3D:
        o3d.visualization.draw_geometries([pcd, mesh_crop], mesh_show_back_face=True)


    """
       Volume Calculation
    
       with the help of PyVista
       https://docs.pyvista.org/
   """
    # Create a Plotter instance
    pv.set_plot_theme("document")

    p = pv.Plotter()
    p.show_bounds()


    # import PyVista Mesh
    v = np.asarray(mesh_crop.vertices)
    f = np.array(mesh_crop.triangles)
    f = np.c_[np.full(len(f), 3), f]
    mesh_pv = pv.PolyData(v, f)

    mesh_pv.points[:, 0] = mesh_pv.points[:, 0] - np.min(mesh_pv.points[:, 0])
    mesh_pv.points[:, 1] = mesh_pv.points[:, 1] - np.min(mesh_pv.points[:, 1])
    mesh_pv.points[:, 2] = mesh_pv.points[:, 2] - np.min(mesh_pv.points[:, 2])

    # Set point color by Scalar Field of coordinate z
    # Mesh will interpolate Point Data
    sf_x = mesh_pv.points[:, 0]
    mesh_pv["elevation"] = sf_x
    if SHOW_PV:
        #   p.add_mesh(mesh_pv, style='wireframe')
        p.add_mesh(mesh_pv)
        p.add_mesh(mesh_pv.copy(), style='wireframe')

    # Manually define base Plane
    h = 0.05
    base = np.array([mesh_pv.center[0]+ h, mesh_pv.center[1], mesh_pv.center[2]])
    direction = o3d_plane[:3]

    trim_surface = pv.Plane(center=base, direction=direction, i_size=2.5, j_size=2)

    if SHOW_PV:
        p.add_mesh(trim_surface, style='wireframe')

    # Using the PyVista - VTK compatibility to apply extrusion to surface filter.
    # PyVista is working already on a native PyVista integration of the functionality
    t = time.time()
    alg = vtk.vtkTrimmedExtrusionFilter()
    alg.SetInputData(0, mesh_pv)
    alg.SetInputData(1, trim_surface)
    alg.SetCappingStrategy(0)  # <-- ensure that the cap is defined by the intersection
    alg.SetExtrusionDirection(h, 0, 0)  # <-- set this with the plane normal
    alg.Update()
    output = pv.core.filters._get_output(alg).compute_normals()
    print("===> DTA Volume Computation [DONE], process time = {:10.4f} seconds".format(time.time() - t))
    print("\n==> Extruded Volume: ", output.volume)

    p.add_mesh(output, color='w')
    p.add_mesh(output.copy(), color='black', style='wireframe', opacity=0.3)

    if SHOW_PV:
        p.show()