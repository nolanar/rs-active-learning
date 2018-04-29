import numpy as np
from functools import reduce, partial

import unittest

class UserGroupModel:

	def __init__(self, init_probs, dist_model):
		self.g = init_probs.shape[0] # number of groups
		self.g_probs = init_probs
		self.dist_model = dist_model

	def expected_utility(self, items, util_f):
		cond_probs = self.dist_model[:, items, :] # [p x q x r_bar] matrix

		# get the P(R_u1 = r_1, ..., R_uq = r_q | G_u = g; \pi)'s as the product 
		# of P(R_ui = r_i | G_u = g; \pi)'s, for all permutations of ratings
		if items.shape[0] > 1:
			cond_probs = np.array(list(map(partial(reduce, np.multiply.outer), cond_probs)))

		# flatten the item and ratings dimensions
		cond_probs = cond_probs.reshape(self.g, -1).T

		# marginalise out groups to get P(R_u1 = r_1, ..., R_uq = r_q; \pi)'s
		r_probs = cond_probs.dot(self.g_probs)

		# Use Bayes' theorem to get P(G_u = g | R_u\pi = r)'s
		cond_probs *= self.g_probs / r_probs[:,None]

		# Return the expected value of the utility over all ratings
		return np.apply_along_axis(util_f, 0, cond_probs).T.dot(r_probs)

class TestUserGroupModel(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(TestUserGroupModel, self).__init__(*args, **kwargs)
		np.random.seed(1)

	def get_test_dist_model(self, group_count=8, item_count=10, rating_count=5):
		dist_m = np.random.randint(8, size=(group_count, item_count, rating_count)).astype(np.float32) + 1
		dist_m /= dist_m.sum(axis=2, keepdims=True) # normalise rating dists
		return dist_m

	def get_test_user_model(self, dist_model, group_count=8):
		init_probs = np.arange(group_count, dtype=np.float32) + 1
		init_probs /= init_probs.sum()
		user_m = UserGroupModel(init_probs, dist_model)
		return user_m

	def check_probs_sum_to_one_when_U_is_ident(self, items):
		dist_m = self.get_test_dist_model()
		user_m = self.get_test_user_model(dist_m)

		U = lambda x: x
		exp_u = user_m.expected_utility(items, U)

		self.assertAlmostEqual(exp_u.sum(), 1.0)

	def test_probs_sum_to_one_when_U_is_ident_with_one_item(self):
		self.check_probs_sum_to_one_when_U_is_ident([2])
	
	def test_probs_sum_to_one_when_U_is_ident_with_several_items(self):
		self.check_probs_sum_to_one_when_U_is_ident([0, 3, 5])

if __name__ == '__main__':
    unittest.main()