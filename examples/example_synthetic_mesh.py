# Author: Lukas Rauch
# Data: 22.05.2022

from turtle import st
from matplotlib.pyplot import show
import open3d as o3d
import numpy as np
from numpy import genfromtxt
import pyvista as pv
import vtk
import time


def synthetic_mesh():
    print("\n=====================================================")
    print("EXAMPLE: Synthetic Mesh Reconstruction")
    print("=====================================================\n")

    pv.set_plot_theme('document')
    p = pv.Plotter()
    p.show_bounds()

    # Import {x,y,z} coordinates from csv
    stl = r"./samples/mesh.stl"
    obj = r"./samples/mesh.obj"
    ply = r"./samples/mesh.ply"


    # reader = pv.get_reader(stl)
    # mesh_stl = reader.read()
    # p.add_mesh(mesh_stl, color='r', point_size=3, opacity=0.4)

    # reader = pv.get_reader(obj)
    # mesh_obj = reader.read()
    # surf = mesh_obj.delaunay_2d()
    # p.add_mesh(surf, color='g', point_size=3, opacity=0.4)


    reader = pv.get_reader(ply)
    mesh_ply = reader.read()
    p.add_mesh(mesh_ply, color='black', style='wireframe')



    # Cutting Plane
    h = -1
    trim_surf = pv.Plane(center=(mesh_ply.center[0], mesh_ply.center[1], h), direction=(0,0,1), i_size=21, j_size=21, i_resolution=21, j_resolution=21)
    p.add_mesh(trim_surf, color='r', opacity=0.3, show_edges=True)

    alg = vtk.vtkTrimmedExtrusionFilter()
    alg.SetInputData(0, mesh_ply)
    alg.SetInputData(1, trim_surf)
    alg.SetCappingStrategy(0) # <-- ensure that the cap is defined by the intersection
    alg.SetExtrusionDirection(0, 0, h) # <-- set this with the plane normal
    alg.Update()
    # output = pyvista.core.filters._get_output(alg)
    output = pv.core.filters._get_output(alg).compute_normals()


    # # Checking Results
    # volume_goal = abs(pyramid.volume - h)
    print("Volume extrusion = {:1.9}." .format(output.volume))
    # assert np.isclose(output.volume, volume_goal)

    p.add_mesh(output, color='w', opacity=0.9, show_edges=True)




    # surf = mesh.delaunay_2d()

    # p.add_mesh(surf)

    p.show()