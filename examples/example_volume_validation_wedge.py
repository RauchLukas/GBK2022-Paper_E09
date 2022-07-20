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
      print("--> Estimate Normals within radius={} and max. nearest_neighbours={} ..." .format(radius, nn))
      pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nn))

      px = 0
      py = 0
      pz = 10000
      print("--> Orient Normals towards position ({}, {}, {})" .format(px, py, pz))
      pcd.orient_normals_towards_camera_location(camera_location=([px, py, pz]))
      if SHOW_O3D:
            o3d.visualization.draw_geometries([pcd, xyz], point_show_normal=True)


      # Poisson Meshing
      d = 6
      print("--> Poisson Meshing with Octree Depth d={} ..." .format(d))
      t = time.time()
      mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=d, width=0, scale=1.1, linear_fit=True, n_threads=-1)
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

      print("----------------------------------------------")



      v = np.asarray(mesh.vertices)
      f = np.array(mesh.triangles)

      mesh = Mesh(v, f)

      plane = Plane([0, 0, 1, 2])
      c_xyz = plane.p_origin

      proj = Projection(mesh=mesh, plane=plane)

      projected_verts = v
      projected_faces = f

      print(projected_verts.shape)
      projected_verts, _ = proj.project_points(mesh.vertices)

      volume = proj.volume_mesh_projected(v, f)
      print("volumes: ", np.sum(volume))

      mesh_area = proj.calculate_triangle_area()
      pv_mesh = pv.PolyData(projected_verts, projected_faces)

      # terrain.cell_data["area"] = mesh_area
      # terrain.set_active_scalars('area')
      #
      # # terrain.plot(show_edges=True)

      pv.set_plot_theme("document")
      plotter = pv.Plotter()


      pv_faces = np.zeros((f.shape[0], 4))
      pv_faces[:] = 4
      pv_faces[1:, :] = f
      pv_faces = pv_faces.reshape(-1)

      mesh = pv.PolyData(v, f)
      #
      # plotter.subplot(0, 0)
      # plotter.add_text("terrain", font_size=10)
      # plotter.add_mesh(terrain, scalars="area", show_edges=True, scalar_bar_args={'title': 'Area'})
      # # plotter.scalar_bar("zzz")
      #
      # plotter.subplot(0, 1)
      # plotter.add_text("volume", font_size=10)
      # plotter.add_mesh(terrain.copy(), scalars=volume, show_edges=True, scalar_bar_args={'title': "Volume"})
      plotter.add_mesh(mesh, show_edges=True)
      #
      # # Display the window
      cpos = plotter.show(full_screen=True)