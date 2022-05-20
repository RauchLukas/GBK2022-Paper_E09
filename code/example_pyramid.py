# Author: Lukas Rauch
# Data: 18.05.2022

from lib2to3.pgen2.grammar import opmap_raw
import open3d as o3d
import numpy as np
from numpy import genfromtxt
import time

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
d = 8
print("--> Poisson Meshing with Octree Depth d={} ..." .format(d))
t = time.time()
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=d, width=0, scale=1.1, linear_fit=True, n_threads=-1)
print("===> Poisson Meshing [DONE], process time = {:10.4f} seconds" .format(time.time() - t))
mesh.compute_triangle_normals()


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



"""
Volume Calculation

with the help of PyVista
https://docs.pyvista.org/
"""
import pyvista as pv
import vtk

# Create a Plotter instance
pv.set_plot_theme("document")

p = pv.Plotter()
p.show_grid()
v = np.asarray(mesh.vertices)
f = np.array(mesh.triangles)
f = np.c_[np.full(len(f), 3), f]
mesh_pv = pv.PolyData(v, f)

# Clip mesh at the X-Y Plane

origin=(0,0,0)
mesh_c = mesh_pv.clip(normal='z', origin=origin, invert=False)
print("\n--> Clipping Poisson Mesh at: ", origin)

# Set point color by Scalar Field of coordinate z
# Mesh will interpolate Point Data
sf_z = mesh_c.points[:, -1]
mesh_c["elevation"] = sf_z
if SHOW_PV:
    #   p.add_mesh(mesh_pv, style='wireframe')
      p.add_mesh(mesh_c, color='w', show_edges=True)


# Manually define base Plane
h = -0.25
base = np.array([mesh_c.center[0], mesh_c.center[1], h])
direction = np.array([0, 0, 1])

trim_surface = pv.Plane(center=base, direction=direction, i_size=2, j_size=2)

if SHOW_PV:
      p.add_mesh(trim_surface, color='r', opacity=0.3)

# Using the PyVista - VTK compatibility to apply extrusion to surface filter.
# PyVista is working already on a native PyVista integration of the functionality
t = time.time()
alg = vtk.vtkTrimmedExtrusionFilter()
alg.SetInputData(0, mesh_c)
alg.SetInputData(1, trim_surface)
alg.SetCappingStrategy(0) # <-- ensure that the cap is defined by the intersection
alg.SetExtrusionDirection(0, 0, h) # <-- set this with the plane normal
alg.Update()
output = pv.core.filters._get_output(alg).compute_normals()
print("===> DTA Volume Computation [DONE], process time = {:10.4f} seconds" .format(time.time() - t))
print("\n==> Extruded Volume: ", output.volume)

p.add_mesh(output, color='gold', show_edges=True, opacity=0.5)

if SHOW_PV:
      p.show()