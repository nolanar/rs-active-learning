import numpy as np
import scipy.sparse as sp
import os.path
import pathlib
from functools import lru_cache # cache large I/O result

from myutils import msg

##
# Utilities for getting rating and group data.
##

######## CONFIG #########
# netflix data
blc_data = '2018-04-03T18:40:29' # features: 20, groups: 8

data_dir = "data/"
figure_dir = "figures/"

# ratings should take integer values 1 to rating_value_count
ratings_file = data_dir + 'netflix_ratings.npz'
rating_value_count = 5 

cache_dir = data_dir + "_cache/"
group_ratings_cache_file =  cache_dir + blc_data + '_group_ratings.npy'

groups_file = data_dir + blc_data + '/P'
V_file = data_dir + blc_data + '/V'
Utilde_file = data_dir + blc_data + '/Utilde'
lam_file = data_dir + blc_data + '/lam'
Rvar_file = data_dir + blc_data + '/Rvar'
P_file = data_dir + blc_data + '/P'
#########################


def read_numpy_file(filename, dtype=np.float32):
	""" Read from numpy file if it exists, otherwise from raw text file """
	with msg(f'Reading "{filename}"'):
		if os.path.isfile(filename + ".npy"): 
			return np.load(filename + ".npy")
		else: 
			return np.loadtxt(open(filename + ".txt", "r"), dtype=dtype)

@lru_cache(maxsize=1)
def get_R():
	""" Returns the ratings matrix in compressed sparse column (csc) format.
	Stores csc matrix to ratings_cache_file for faster loading in future.
	Cached result to allow single load on multiple calls. 
	"""
	filename = DataReader.ratings_file
	if os.path.isfile(filename):
		with msg(f'Loading rating matrix from "{filename}"'):
			return sp.load_npz(filename)
	else:
		raise RuntimeError(f'"{filename}" does not exist. Use "netflix_data.py" to generate it.')

@lru_cache(maxsize=1)
def get_groups():
	""" Returns the groups as a list of numpy arrays.
	Cached result to allow single load on multiple calls.
	"""
	P = DataReader.get_P()
	with msg(f'Converting to list of user indexes by group'):
		indexes = np.arange(P.shape[1])
		return [indexes[group] for group in P.astype(bool)]

@lru_cache(maxsize=1)
def group_sizes():
	return DataReader.get_P().sum(axis=1)

def number_of_users():
	return DataReader.get_P().shape[1]

def number_of_groups():
	return DataReader.get_P().shape[0]

def group_distribution():
	return DataReader.group_sizes() / DataReader.number_of_users()

@lru_cache(maxsize=1)
def get_Rtilde():
	V = DataReader.read_numpy_file(DataReader.V_file)
	Utilde = DataReader.read_numpy_file(DataReader.Utilde_file)
	return np.dot(Utilde.T, V)

@lru_cache(maxsize=1)
def get_Rvar():
	return DataReader.read_numpy_file(DataReader.Rvar_file)

@lru_cache(maxsize=1)
def get_lam():
	""" number of ratings for each item by each group """
	return DataReader.read_numpy_file(DataReader.lam_file)

@lru_cache(maxsize=1)
def get_P():
	filename = DataReader.P_file
	if os.path.isfile(filename + '.npz'):
		with msg(f'Reading "{filename}.npz"'):
			return sp.load_npz(filename + '.npz').toarray()
	elif os.path.isfile(filename + '.npy'):
		with msg(f'Reading "{filename}.npy"'):
			return sp.load(filename + '.npy')

@lru_cache(maxsize=1)
def get_group_ratings():
	"""
	Returns a [p, m, r] array containing the ratings given by each group to each item 
	p = number of groups
	m = number of items
	r = number of rating values
	"""
	cachefile = DataReader.group_ratings_cache_file
	if os.path.isfile(cachefile):
		with msg(f'Reading group ratings from "{cachefile}"'):
			return np.load(cachefile)

	with msg('Getting group ratings'):
		R = DataReader.get_R()
		P = DataReader.get_groups()
		number_of_groups = DataReader.number_of_groups()
		item_count = R.shape[1]
		rating_count = DataReader.rating_value_count

		ratings = np.zeros((number_of_groups, item_count, rating_count), dtype=np.float32)
		with msg(f'Calculating rating counts'):
			for group_n, group in enumerate(P):
				for rating_index in range(rating_count):
					rating = rating_index + 1
					ratings[group_n, :, rating_index] = (R[group] == rating).sum(axis=0)

		with msg(f'Saving group ratings to "{cachefile}"'):
			np.save(cachefile, ratings)

		return ratings
