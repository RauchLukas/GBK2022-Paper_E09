# Author: Lukas Rauch
# Data: 11.05.2022

import open3d as o3d
import numpy as np
from numpy import genfromtxt
import time

print("\n=====================================================")
print("EXAMPLE: synthetic Point Cloud Surface Reconstruction")
print("=====================================================\n")

# Set Visualization Toggle
SHOW_O3D = True
SHOW_PV = True

# Import {x,y,z} coordinates from csv
path = "./samples/pyramid.csv"
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
if SHOW_O3D:
      o3d.visualization.draw_geometries([pcd, xyz])


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
if SHOW_O3D:
      o3d.visualization.draw_geometries([pcd, xyz], point_show_normal=True)


# Poisson Meshing
d = 8
print("--> Poisson Meshing with Octree Depth d={} ..." .format(d))
t = time.time()
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=d, linear_fit=True)
print("===> Poisson Meshing [DONE], process time = {:10.4f} seconds" .format(time.time() - t))
mesh.compute_triangle_normals()

# Crop the Mesh within the bounding Box
bbox.min_bound = np.array([0, 0, 0])
bbox.max_bound = np.array([2, 2, 1])

print(bbox)
mesh = mesh.crop(bbox)

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
import pyvista as pv
import vtk

# Create a Plotter instance
p = pv.Plotter()
p.show_bounds()

v = np.asarray(mesh.vertices)
f = np.array(mesh.triangles)
f = np.c_[np.full(len(f), 3), f]
mesh_pv = pv.PolyData(v, f)

print(mesh_pv.bounds)

# mesh_pv = mesh_pv.clip_box(bounds=(0, 2, 0, 2, 0, 1), invert=False)
# mesh_pv.triangulate(inplace=True)

# Set point color by Scalar Field of coordinate z
# Mesh will interpolate Point Data
sf_z = mesh_pv.points[:, -1]
mesh_pv["elevation"] = sf_z
# if SHOW_PV:
#       p.add_mesh(mesh_pv, style='wireframe')


# Manually define base Plane

base = np.array([1, 1, -1])

plane = pv.Plane(base, (0, 0, -1), i_size=3, j_size=3)
plane.triangulate(inplace=True)

if SHOW_PV:
      p.add_mesh(plane, color='r', opacity=0.3)


alg = vtk.vtkTrimmedExtrusionFilter()
alg.SetInputData(0, mesh_pv)
alg.SetInputData(1, plane)
alg.SetCappingStrategy(0) # <-- ensure that the cap is defined by the intersection
alg.SetExtrusionDirection(0, 0, -1.0) # <-- set this with the plane normal
alg.Update()
output = pv.core.filters._get_output(alg)
print("Output Volume: ", output.volume)
# p.add_mesh(output)



# Extrude plane to quad
# points_z = np.asarray(pcd.points)
# height = np.amax(points_z[:, -1], axis=0) - base[-1]
# extrude = plane.extrude((0, 0, height), capping=False)

extrude = mesh_pv.extrude((0.0, 0, -2), capping=False)
extrude.triangulate(inplace=True)
#
# print("extr. vol: ", extrude.volume)
#
# print("ext: ", extrude.volume)
#
# if SHOW_PV:
#       p.add_mesh(extrude, color='w', opacity=0.9)

# # mesh_pv.flip_normals()
volume = extrude.clip_surface(plane)
# #
# #
# volume = extrude.boolean_difference(plane)
# # volume = extrude.boolean_difference(plane.triangulate(inplace=True))
print("==> Volume = {}" .format(volume.volume))
if SHOW_PV:
      p.add_mesh(volume, color='lightgreen', opacity=0.7, show_edges=True)


if SHOW_PV:
      p.show()





