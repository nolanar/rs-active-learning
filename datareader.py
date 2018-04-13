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

	# ratings should take integer values 1 to rating_value_count
	ratings_file = data_dir + 'netflix_ratings.npz'
	rating_value_count = 5 

	cache_dir = data_dir + "_cache/"
	nym_stats_cache_file = cache_dir + blc_data + '_nym_stats_v2.npy'
	group_rating_dists_cache_file =  cache_dir + blc_data + '_group_rating_dists{}.npy'

	nyms_file = data_dir + blc_data + '/P'
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
		P = DataReader.get_P()
		with msg(f'Converting to list of user indexes by group'):
			indexes = np.arange(P.shape[1])
			return [indexes[group] for group in P.astype(bool)]

	@lru_cache(maxsize=1)
	def nym_sizes():
		return DataReader.get_P().sum(axis=1)

	def number_of_users():
		return DataReader.get_P().shape[1]

	def nym_count():
		return len(DataReader.get_nyms())

	@lru_cache(maxsize=1)
	def get_nym_stats():
		""" Returns statistics about rating distributions of all items for each nym
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
	def get_group_rating_distributions(reg=10):
		"""
		Returns a [p, m, r] array containing distribution of ratings for each item by group
		p = number of groups
		m = number of items
		r = rating
		"""
		cachefile = DataReader.group_rating_dists_cache_file.format(f'_reg{reg}' if reg else '')
		if os.path.isfile(cachefile):
			with msg(f'Reading distribution of ratings for each item per group from "{cachefile}"'):
				return np.load(cachefile)

		with msg('Getting distribution of ratings for each item and group'):
			R = DataReader.get_ratings()
			P = DataReader.get_nyms()
			group_count = DataReader.nym_count()
			item_count = R.shape[1]
			rating_count = DataReader.rating_value_count

			dists = np.zeros((group_count, item_count, rating_count), dtype=np.float32)
			with msg(f'Calculating distributions'):
				for group_n, group in enumerate(P):
					for rating_index in range(rating_count):
						rating = rating_index + 1
						dists[group_n, :, rating_index] = (R[group] == rating).sum(axis=0)
			
			# Add small value to regularise low popularity items
			if reg: dists += reg/rating_count

			with msg('Normalising distributions'):
				dists /= dists.sum(axis=2, keepdims=True)
				# nan's imply 0 ratings by group on item, so give equal distribution 
				dists[np.isnan(dists)] = 1.0 / rating_count

			with msg(f'Saving distribution of ratings to "{cachefile}"'):
				np.save(cachefile, dists)

			return dists
