import numpy as np
import scipy.sparse as sp
import os.path
import pathlib

from myutils import msg

from functools import lru_cache # cache large I/O result

class DataReader:
	""" Utility class for getting rating and nym data """
	
	######## CONFIG #########
	# netflix data
	blc_data = '2018-04-03T18:40:29' # features: 20, nyms: 8

	data_dir = "data/"
	figure_dir = "figures/"

	ratings_file = data_dir + 'netflix_ratings.npz'

	cache_dir = data_dir + "_cache/"
	nym_stats_cache_file = cache_dir + blc_data + '_nym_stats_v2.npy'

	nyms_file = data_dir + blc_data + '/P'
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
		filename = DataReader.ratings_file
		if os.path.isfile(filename):
			with msg(f'Loading rating matrix from "{filename}"'):
				return sp.load_npz(filename)
		else:
			raise RuntimeError(f'"{filename}" does not exist. Use "netflix_data.py" to generate it.')

	@lru_cache(maxsize=1)
	def get_nyms():
		""" Returns the nyms as a list of numpy arrays.
		Cached result to allow single load on multiple calls.
		"""
		filename = DataReader.nyms_file
		with msg(f'Reading nyms from "{filename}"'), open(filename, 'r') as f: 
			nyms_raw = np.loadtxt(f, delimiter=',', dtype=int)
			# parse into list of nyms
			nym_count = nyms_raw[:,1].max() + 1
			return [ nyms_raw[:,0][nyms_raw[:,1]==nym_n] for nym_n in range(0, nym_count) ]
	
	def nym_count():
		return len(DataReader.get_nyms())

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
			stats = np.zeros((len(nyms), ratings.shape[1], 4), dtype=np.float32)
			for nym_n, nym in enumerate(nyms):
				with msg(f'Getting nym #{nym_n} stats'):
					for i, items in enumerate(ratings[nym].T):
						data = items.data
						stats[nym_n, i, 0] = i
						stats[nym_n, i, 1] = data.mean() if len(data) > 0 else 0
						stats[nym_n, i, 2] = data.var() if len(data) > 0 else 0
						stats[nym_n, i, 3] = len(data)
			with msg(f'Saving nym stats to "{filename}"'):
				np.save(filename, stats)
		return stats
