# Author: Lukas Rauch
# Data: 18.05.2022

import open3d as o3d
import numpy as np
from numpy import genfromtxt
import pyvista as pv
import vtk
import time

def mesh_quality():
      
      print("\n=====================================================")
      print("EXAMPLE: Computes the scaled Jacobian mesh quality")
      print("=====================================================\n")

      # Set Visualization Toggle
      SHOW_O3D = False
      SHOW_PV = True

      # Import {x,y,z} coordinates from csv
      path = r"./samples/witch_hat.csv"
      data = genfromtxt(path, delimiter=',')
      print("--> Importing: ", path)

      # Create open3D Point Cloud Database
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(data)
      print("===> ", pcd)

      # Create a Coordinate Frame
      xyz = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

      # Normals
      radius = 0.1
      nn = 30
      print("--> Estimate Normals within radius={} and max. nearest_neighbours={} ..." .format(radius, nn))
      pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nn))

      px = 0
      py = 0
      pz = -10000
      print("--> Orient Normals towards position ({}, {}, {})" .format(px, py, pz))
      pcd.orient_normals_towards_camera_location(camera_location=([px, py, pz]))

      # Poisson Meshing
      d = 8
      print("--> Poisson Meshing with Octree Depth d={} ..." .format(d))
      t = time.time()
      mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=d, width=0, scale=2, linear_fit=True, n_threads=-1)
      print("===> Poisson Meshing [DONE], process time = {:10.4f} seconds" .format(time.time() - t))

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


      """
      Volume Calculation

      with the help of PyVista
      https://docs.pyvista.org/
      """

      # Create a Plotter instance
      p = pv.Plotter()
      p.show_bounds()

      v = np.asarray(mesh.vertices)
      f = np.array(mesh.triangles)
      f = np.c_[np.full(len(f), 3), f]
      mesh_pv = pv.PolyData(v, f)


      # Clip mesh
      mesh_pv = mesh_pv.clip(normal='z', origin=(0,0,0), invert=False)
      p.add_mesh(mesh_pv, style='wireframe')

      t = time.time()
      qual = mesh_pv.compute_cell_quality(quality_measure='scaled_jacobian')
      print("\n===> Computing Jacobian mesh quality [DONE], process time = {:10.4f} seconds" .format(time.time() - t))

      p.add_mesh(qual, opacity=0.99)

      p.show()

