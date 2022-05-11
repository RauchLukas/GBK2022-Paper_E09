# Author: Lukas Rauch
# Data: 11.05.2022


import open3d as o3d
import numpy as np
from numpy import genfromtxt
import time

print("\n=====================================================")
print("EXAMPLE: synthetic Point Cloud Surface Reconstruction")
print("=====================================================\n")


# Import {x,y,z} coordinates from csv
path = "../samples/pyramid.csv"
data = genfromtxt(path, delimiter=',')
print("--> Importing: ", path)

# Create open3D Point Cloud Database
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)
print("===> ", pcd)

# Get the bounding Box
bbox = pcd.get_axis_aligned_bounding_box()
bbox.color = (1, 0, 0)

# Create a Coordinate Frame
xyz = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

# Visualize Point Cloud
# o3d.visualization.draw_geometries([pcd, xyz])

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
o3d.visualization.draw_geometries([pcd, xyz], point_show_normal=True)


# Poisson Meshing
d = 8
print("--> Poisson Meshing with Octree Depth d={} ..." .format(d))
t = time.time()
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=d)
print("===> Poisson Meshing [DONE], process time = {:10.4f} seconds" .format(time.time() - t))
mesh.compute_triangle_normals()

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
import pyvista as pv



