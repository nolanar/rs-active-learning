import numpy as np
from datareader import DataReader

class User:
	"""
	Class to represent a synthetic user
	"""

	def __init__(self, group, dists):
		self.group = group
		group_d = dists[group]

		rating_chooser = lambda p: np.random.choice(DataReader.rating_value_count, p=p) + 1
		self.ratings = np.apply_along_axis(rating_chooser, 1, group_d)
