import numpy as np
from functools import reduce, partial

import unittest

class UserGroupModel:

	def __init__(self, init_probs, dist_model):
		self.g = init_probs.shape[0] # number of groups
		self.g_probs = init_probs
		self.dist_model = dist_model

	def expected_utility(self, items, util_f):
		probs = self.dist_model[:, items, :] # [p x q x r_bar] matrix

		# Get the product of all perms of ratings as a [(r_bar ^ q) x p] matrix
		# i.e. get the P(R_u1 = r_1, ..., R_uq = r_q | G_u = g; \pi)'s, with g as column
		cond_probs = np.array(list(map(partial(reduce, np.outer), probs))).reshape(self.g, -1).T

		# marginalise out groups to get P(R_u1 = r_1, ..., R_uq = r_q; \pi)'s
		r_probs = cond_probs.dot(self.g_probs)

		# Use Bayes' theorem to get P(G_u = g | R_u\pi = r)'s
		cond_probs *= self.g_probs / r_probs[:,None]

		# Return the expected value of the utility over all ratings
		return np.apply_along_axis(util_f, 0, cond_probs).T.dot(r_probs)

class TestUserGroupModel(unittest.TestCase):

	def test_probs_sum_to_one_when_U_is_ident(self):
		np.random.seed(1)

		group_count, item_count, rating_count = 8, 10, 5
		dist_m = np.random.randint(8, size=(group_count, item_count, rating_count)).astype(np.float32) + 1
		dist_m /= dist_m.sum(axis=2, keepdims=True) # normalise rating dists

		init_probs = np.arange(group_count, dtype=np.float32) + 1
		init_probs /= init_probs.sum()
		user_m = UserGroupModel(init_probs, dist_m)

		items = [0, 2, 5]
		U = lambda x: x
		exp_u = user_m.expected_utility(items, U)
		
		self.assertAlmostEqual(exp_u.sum(), 1.0)
	
if __name__ == '__main__':
    unittest.main()