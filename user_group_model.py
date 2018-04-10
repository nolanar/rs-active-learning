import numpy as np
from functools import reduce, partial

class UserGroupModel:

	def __init__(self, init_probs, dist_model):
		self.groups = np.arange(len(init_probs))
		self.group_probs = init_probs
		self.dist_model = dist_model

	def lookahead(self, items):
		probs = self.dist_model[:, items, :] # p x q x r_bar matrix

		# Get the product of all perms of ratings as an ndarray
		# i.e. P(R_u1 = r_1, ..., R_uq = r_q | G_u = g; \pi)'s
		prob_r_g = np.array(list(map(partial(reduce, np.outer), probs)))

		# sum over groups to get P(R_u1 = r_1, ..., R_uq = r_q; \pi)'s
		prob_r = np.inner(self.group_probs, prob_r_g.T).T

		# elementwise divide prob_r_g for each group by prob_r
		for g in self.groups: prob_r_g[g] /= prob_r

		# sum these to get group_probs after items rated
		return self.group_probs * np.array([ps.sum() for ps in prob_r_g])

def test():
	np.random.seed(1)

	group_count, item_count, rating_count = 2, 10, 2
	dist_m = np.random.randint(8, size=(group_count, item_count, rating_count)).astype(np.float32) + 1
	dist_m /= dist_m.sum(axis=2, keepdims=True) # normalise rating dists

	init_probs = np.arange(group_count, dtype=np.float32) + 1
	init_probs /= init_probs.sum()
	user_m = UserGroupModel(init_probs, dist_m)

	items = [0]
	la = user_m.lookahead(items)
	print(la)
	print(la.sum())
	
if __name__ == '__main__':
	test()