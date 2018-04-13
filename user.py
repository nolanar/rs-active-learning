import numpy as np
from datareader import DataReader
from myutils import msg

def synthesise_user_data(dists, group_sizes):
	"""
	Synthesise user data using a distribution of ratings for each item by group.
	group_sizes(list) = number of users to be generated per group
	returns 2D ratings array, and groups list corresponding to which group each user belongs to
	"""
	with msg(f'Building {np.sum(group_sizes)} synth users each with {dists.shape[1]} ratings'):
		ratings = []
		r_dim = DataReader.rating_value_count
		for group_size, group in zip(group_sizes, dists):
			if group_size != 0:
				rating_chooser = lambda p: np.random.choice(r_dim, size=group_size, p=p) + 1
				ratings.append(np.stack(np.apply_along_axis(rating_chooser, 1, group).T))
		ratings = np.vstack(ratings)

		groups = np.concatenate([np.full(group_size, group) for group, group_size in enumerate(group_sizes)])

		perm = np.random.permutation(ratings.shape[0])
		return ratings[perm], groups[perm]

def demo_synthesise_user_data():
	np.random.seed(0)
	item_count = None
	dists = DataReader.get_group_rating_distributions()[:, :item_count, :]

	group_sizes=np.random.randint(10, 100, size=DataReader.nym_count())
	ratings, groups = synthesise_user_data(dists, group_sizes=group_sizes)

if __name__ == '__main__':
	demo_synthesise_user_data()