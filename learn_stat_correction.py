import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

import argparse

def trunc_mean_and_var(a, b, mean, var):
	sd = np.sqrt(var)
	cdf_a, cdf_b = norm.cdf((a, b), mean, sd)
	pdf_a, pdf_b = norm.pdf((a, b), mean, sd)

	t_mean = ((mean - b)*cdf_b - var*pdf_b)- ((mean - a)*cdf_a - var*pdf_a) + b
	
	t_var = ((mean**2 + var - b**2)*cdf_b - var*(mean + b)*pdf_b) \
	      - ((mean**2 + var - a**2)*cdf_a - var*(mean + a)*pdf_a) \
	      + b**2 - t_mean**2

	return t_mean, t_var

def plot_sliding_mean(option='mean'):
	a, b = 1, 5
	var = 1

	points = 32
	xs = np.linspace(0, 6, num=points)
	map_f = lambda x: trunc_mean_and_var(a, b, x, var)
	y_mean, y_var = map(np.array, zip(*map(map_f, xs)))
	
	plt.axvline(a, linestyle='dotted', c='g')
	plt.axvline(b, linestyle='dotted', c='g')

	if option is 'var':
		plt.plot(xs, y_var)
		plt.plot(xs, np.zeros(points) + var)
	elif option is 'mean':
		plt.plot(xs, y_mean)
		plt.plot(xs, xs)
	
	plt.show()

def plot_sliding_var(option='var'):
	a, b = 1, 5
	mean = 2

	points = 32
	xs = np.linspace(0.01, 5, num=points)
	map_f = lambda x: trunc_mean_and_var(a, b, mean, x)
	y_mean, y_var = map(np.array, zip(*map(map_f, xs)))
	
	if option is 'var':
		plt.plot(xs, y_var, label='trunc var')
		plt.plot(xs, xs, label='true var')
	elif option is 'mean':
		plt.plot(xs, y_mean, label='trunc mean')
		plt.plot(xs, np.zeros(points) + mean, label='true mean')
	
	plt.title(f'mean {mean}')
	plt.legend()
	plt.show()

def get_mesh(option='mean'):
	a, b = 1, 5
	padding = 10
	x_points = 256
	x = np.linspace(a - padding, b + padding, num=x_points)

	v_low, v_high = 0.001, 15
	v_points = 256
	v = np.linspace(v_low, v_high, num=v_points)

	xx, vv = np.meshgrid(x, v)
	points = np.column_stack((xx.flatten(), vv.flatten()))

	map_f = lambda p: trunc_mean_and_var(a, b, p[0], p[1])
	t_x, t_v = map(np.array, zip(*map(map_f, points)))

	# grid_x, grid_y = np.meshgrid(x, v)
	# grid_x, grid_y = np.meshgrid(np.linspace(1, 5, num=256), v)
	grid_x, grid_y = np.meshgrid(np.linspace(1, 5, num=256), np.linspace(np.min(t_v), np.max(t_v), num=256))

	# grid_z = griddata(points, t_v, (grid_x, grid_y), method='cubic')
	# grid_z = griddata(np.column_stack((t_x, points[:,1])), t_v, (grid_x, grid_y), method='linear')
	col = 1 if option is 'variance' else 0
	grid_z = griddata(np.column_stack((t_x, t_v)), points[:,col], (grid_x, grid_y), method='linear')
	print(grid_z)
	

	# for i in range(10, 256, 10):
	i = 90
	x = np.linspace(1, 5, num=256)
	y = grid_z[i]
	x = x[~np.isnan(y)]
	y = y[~np.isnan(y)]
	
	plt.title(f'Calculated variance = {grid_y[i, 0]}')
	plt.xlabel('calculated mean')
	plt.ylabel(f'corrected {option}')
	plt.plot(x, y)
	plt.show()


	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(grid_x, grid_y, grid_z)

	plt.show()

get_mesh()

################ EXPERIMENTAL ###################

def get_fit(option='mean', var=2, show=True):	
	a, b = 1, 5

	points = 250
	xs = np.linspace(-4, 10, num=points)
	map_f = lambda x: trunc_mean_and_var(a, b, x, var)
	y_mean, y_var = map(np.array, zip(*map(map_f, xs)))

	if option is 'mean':
		func = lambda x, sd: norm.cdf(x, (a+b)/2, sd) * (b - a) + 1

		popt, pcov = curve_fit(func, xs, y_mean)

		if show:
			plt.plot(xs, xs, label='true mean')
			plt.plot(xs, y_mean, label='trunc mean')
			plt.plot(xs, func(xs, *popt), label='fit')
			plt.plot(xs, xs * y_mean / func(xs, *popt), label='corrected')
			plt.legend()
			plt.show()
		else:
			return popt

	elif option is 'var':
		_, scale = map_f((a+b)/2)
		def func(x, vpdf, sm):
			return norm.pdf(x, (a+b)/2, np.sqrt(vpdf)) * sm#* scale / norm.pdf(0, scale=np.sqrt(vpdf))
				# * norm.cdf(x, a, np.sqrt(vcdf))

		popt, pcov = curve_fit(func, xs, y_var)
		print(popt)

		if show:
			plt.plot(xs, np.zeros(points) + var, label='true var')
			plt.plot(xs, y_var, label='trunc var')
			plt.plot(xs, func(xs, *popt), label='fit')
			plt.plot(xs, var * y_var / func(xs, *popt), label='corrected')
			plt.legend()
			plt.show()
		else:
			return popt

def test_plot():
	mean = 0
	var = 1

	# f = lambda x, mean, var: norm.pdf(xs, mean, np.sqrt(var))
	f = lambda x, mean, var: np.e ** (-(abs(x - mean))**100 / (2 * var**2))

	xs = np.linspace(-3, 3, num=100)
	ys = f(xs, mean, var)
	hs = (xs > -1) & (xs < 1)

	l = 0.5
	# plt.plot(xs, l*ys + (1-l)*hs)
	plt.plot(xs, ys)
	plt.show()

def plot_powers():
	# a, b = 1, 5
	# var = 1

	# xs = np.linspace(-1, 7, num=100)
	# map_f = lambda x: trunc_mean_and_var(a, b, x, var)
	# y_mean, y_var = map(np.array, zip(*map(map_f, xs)))


	xs = np.linspace(0.1, 2.5, num=100)
	ys = []
	for v in xs:
		ys.append(get_fit(var=v, show=False)[0])

	# func = lambda x, b: 2/np.sqrt(x) + b*x # for p
	# func = lambda x, a, p, b: -a/x**p + 2 # for sm
	func = lambda x, a, b: a * x + b

	popt, pcov = curve_fit(func , xs, ys)
	print(popt)
	plt.plot(xs, func(xs, *popt), label='fit')
	plt.legend

	plt.plot(xs, ys)
	plt.show()
