import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
import pyvista as pv
import numpy as np
from time import time

from scripts.mesh import Mesh
from scripts.projection_plane import Plane, Projection

from numpy.random import randint, random


def example_volume_blob_stats():

	# Prepate Data
	columns = ["run", "nx", "ny", "n_cells", "area", "area_goal", "volume", "volume_goal", "eps", "error", "calc_time"]
	df = pd.DataFrame(columns=columns)

	runs = np.arange(11, 102, 10)
	runs = np.concatenate((runs, np.arange(121, 202, 20)))
	runs = np.concatenate((runs, np.arange(251, 402, 50)))

	for i, dxy in enumerate(runs[:5]):
		s = time()

		x0, x1 = 0, 1
		y0, y1 = 0, 1
		d = 8

		# Function to evaluate
		function_z = lambda x, y: d * (1 - x) * x * (1 - y) * y

		# reference value
		volume_goal = integrate.dblquad(function_z, y0, y1, x0, y1)[0]

		x = np.linspace(0., 1., dxy)
		y = np.linspace(0., 1., dxy)
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
		area = np.sum(projection.calculate_triangle_area())
		volume = np.sum(projection.volume_mesh_projected(vertices, faces))

		eps = abs(volume - volume_goal)
		error = abs(100 * eps / volume)

		calc_time = time()-s

		data = {"run": i,
		       "nx": dxy,
		       "ny": dxy,
		        "n_cells": mesh.n_faces,
		        "area": area,
		        "area_goal": None,
		        "volume": volume,
		        "volume_goal": volume_goal,
		        "eps": eps,
		        "error": error,
		        "calc_time": calc_time,
		        }
		df.loc[i] = data

		print("[INFO] Run {}, dnx = dny {}, time: {:.5f} sec." .format(i, dxy, calc_time))


	print(df)

	fig, ax = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)

	# gca stands for 'get current axis'

	df.plot(kind='line', x='run', y='calc_time', ylabel='calculation time [sec.]', color='black', ax=ax[0, 0])
	df.plot(kind='line', x='run', y='n_cells', ylabel='number faces', color='green', ax=ax[0, 1])
	df.plot(kind='line', x='run', y='area', ylabel='area [unit^2]', color='blue', ax=ax[1, 0])
	df.plot(kind='line', x='run', y='error', ylabel='error [%]', color='red', ax=ax[1, 1], logy=True)

	plt.show()

	#
	#
