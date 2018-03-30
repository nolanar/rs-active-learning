import numpy as np
import scipy.sparse as sp
import os.path
import pathlib

from myutils import msg

from functools import lru_cache # cache large I/O result

class DataReader:
	""" Utility class for getting rating and nym data """
	
	######## CONFIG #########
	data_dir = "data/"
	figure_dir = "figures/"

	users_file = data_dir + 'users_big'
	items_file = data_dir + 'movies_big'
	ratings_file = data_dir + 'ratings_big'

	cache_dir = data_dir + "_cache/"
	ratings_cache_file = cache_dir + 'ratings.npz'
	nym_stats_cache_file = cache_dir + 'nym_stats.npy'

	nyms_file = data_dir + 'P'
	nym_count = 8 # number of nyms in 'data/P'
	#########################


	def read_numpy_file(filename, dtype=np.float32):
		""" Read from numpy file if it exists, otherwise from raw text file """
		with msg(f'Reading "{filename}"'):
			if os.path.isfile(filename + ".npy"): 
				return np.load(filename + ".npy")
			else: 
				return np.loadtxt(open(filename + ".txt", "r"), dtype=dtype)

	@lru_cache(maxsize=1)
	def get_ratings():
		""" Returns the ratings matrix in compressed sparse column (csc) format.
		Stores csc matrix to ratings_cache_file for faster loading in future.
		Cached result to allow single load on multiple calls. 
		"""
		filename = DataReader.ratings_cache_file
		if os.path.isfile(filename):
			with msg(f'Loading rating matrix from "{filename}"'):
				ratings = sp.load_npz(filename)
		else:
			f_ratings = DataReader.read_numpy_file(DataReader.ratings_file)
			f_users = DataReader.read_numpy_file(DataReader.users_file, dtype=int)
			f_items = DataReader.read_numpy_file(DataReader.items_file, dtype=int)
			with msg('Forming rating matrix'):
				ratings = sp.coo_matrix((f_ratings, (f_users, f_items)), dtype=np.float32).tocsc()
			with msg(f'Saving rating matrix to "{filename}"'):
				sp.save_npz(filename, ratings)
		return ratings

	@lru_cache(maxsize=1)
	def get_nyms():
		""" Returns the nyms as a list of numpy arrays.
		Cached result to allow single load on multiple calls.
		"""
		filename = DataReader.nyms_file
		with msg(f'Reading nyms from "{filename}"'), open(filename, 'r') as f: 
			nyms_raw = np.loadtxt(f, delimiter=',', dtype=int)
			# parse into list of nyms
			return [ nyms_raw[:,0][nyms_raw[:,1]==nym_n] for nym_n in range(0, DataReader.nym_count) ]
	
	@lru_cache(maxsize=1)
	def get_nym_stats():
		""" Returns statistics about rating distributions of all items for each nym,
		as a 3d numpy array [nym number, item number, <stat>] (type np.float32),
		where <stat> index
		  0 : item index
		  1 : distribution mean
		  2 : distribution variance
		  3 : number of ratings
		Cached result to allow single load on multiple calls.
		"""
		filename = DataReader.nym_stats_cache_file
		if os.path.isfile(filename):
			with msg(f'Reading nym stats from "{filename}"'):
				stats = np.load(filename)
		else:
			ratings = DataReader.get_ratings()
			nyms = DataReader.get_nyms()
			stats = np.zeros((len(nyms), ratings.shape[1], 3), dtype=np.float32)
			for nym_n, nym in enumerate(nyms):
				with msg(f'Getting nym #{nym_n} stats'):
					for i, items in enumerate(ratings[nyms[nym_n]].T):
						data = items.data
						stats[nym_n, i, 0] = data.var() if len(data) > 0 else 0
						stats[nym_n, i, 1] = len(data)
						stats[nym_n, i, 2] = i
			with msg(f'Saving nym stats to "{filename}"'):
				np.save(filename, stats)
		return stats
