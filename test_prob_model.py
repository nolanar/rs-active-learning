import numpy as np
import random
import matplotlib.pyplot as plt

from myutils import msg
from datareader import DataReader as Data
from user import synthesise_user_data

def get_group_probs(init_probs, ratings, prob_R_G): #, prob_S_G):
	post = init_probs.copy()

	if type(ratings) is np.ndarray:
		ratings = enumerate(ratings)
	else:
		ratings = ratings.tocoo()
		ratings = zip(ratings.col, ratings.data)

	for item_n, rating in ratings:
		post *= prob_R_G[:, item_n, int(rating-1)] # * prob_S_G[:, item_n]
		post /= post.sum() # normalise probs
	return post

def probs_vs_true_groups_real_data(test_user_n=1000, print_output=False, print_msg=True):
	with msg("Getting data"):
		R = Data.get_ratings()
		P = Data.get_P()
		P_list = np.arange(P.shape[0]).dot(P)
		init_probs = Data.nym_sizes() / Data.number_of_users()
		prob_R_G = Data.get_group_rating_distributions() # P(R=r|G=g)'s
		## for prob_S_G: (BROKEN)
		# lam = Data.get_lam()
		# prob_S_G = lam / lam.sum(axis=1, keepdims=True) # probability of item being picked by group member

	with msg('Getting count of correct group probs', enabled=print_msg):
		correct_count = 0
		for user_n, user_ratings in enumerate(R[:test_user_n]):
			post = get_group_probs(init_probs, user_ratings, prob_R_G) #, prob_S_G)

			prob_group = post.argmax()
			actual_group = P_list[user_n]
			if prob_group == actual_group: correct_count += 1
				
			if print_output:
				max_prob = post.max()
				user_n_width = int(np.log10(test_user_n)+1)
				print(f'user {user_n:{user_n_width}}: actual {actual_group}, model {prob_group}, {prob_group == actual_group}, prob {np.array(max_prob):.{3}}')		
	
	print(f'acccuracy: {correct_count / test_user_n} ({correct_count} of {test_user_n})')

def probs_vs_true_groups_trial_users(users=1000, samples=1000, items_per_sample=5, sample_from_items=None, print_stats=True, dist_reg=None):
	"""
	For each trial, the group of a synthesised user is predicted using several samplings of items. 
	A prediction accuracy is found from this for each trial, and summary statistics displayed.
	"""
	with msg("Generating synthetic data", enabled=print_stats):
		group_pop_dist = Data.nym_sizes() / Data.number_of_users()
		group_sizes = (group_pop_dist * users).astype(int)
		remainder = users - group_sizes.sum()
		group_sizes[np.argpartition(group_pop_dist, -remainder)[-remainder:]] +=1
		
		if dist_reg is None:
			dists = Data.get_group_rating_distributions()
		else:
			dists = Data.get_group_rating_distributions(reg=dist_reg)

		if sample_from_items is None: sample_from_items = range(dists.shape[1])
		itemss = np.array([random.sample(sample_from_items, items_per_sample) for _ in range(samples)])
		itemss.sort(axis=1)
		all_items = np.unique(itemss)
		R, P_list = synthesise_user_data(dists[:, all_items, :], group_sizes)

		init_probs = Data.nym_sizes() / Data.number_of_users()

	with msg(f'Calculating acccuracy over {users} users and {samples} samples, and {items_per_sample} items per sample', enabled=print_stats):
		correct_preds = np.zeros((users, samples), dtype=bool)
		for user, user_ratings, group in zip(range(users), R, P_list):
			actual_group = P_list[user]
			for sample, items in enumerate(itemss):
				ratings = user_ratings[np.isin(all_items, items)]
				post = get_group_probs(init_probs, ratings, dists[:,items])
				correct_preds[user, sample] = post.argmax() == actual_group

	if print_stats:
		correct_pred_percents = correct_preds.mean(axis=0)
		if correct_pred_percents.shape[0] < 10:
			print(f'\nAccuracy of sample(s): {correct_pred_percents}')
		else:
			min_per = correct_pred_percents.min()
			max_per = correct_pred_percents.max()
			mean_per = correct_pred_percents.mean()
			median_per = np.median(correct_pred_percents)
			print(f'\nAccuracy of samples ({samples}), across users ({users}):')
			print(f'min:\t{min_per:#.3}\nmax:\t{max_per:#.3}\nmean:\t{mean_per:#.3}\nmedian:\t{median_per:#.3}')

	return correct_preds

def probs_vs_true_groups_trial_users_most_ratings(n_most=1000):
	"""
	Sample items from the n most rated items only, to use with probs_vs_true_groups_trial_users.
	"""
	nym_stats = Data.get_nym_stats()
	item_rating_counts = nym_stats[:,:,3].astype(int).sum(axis=0)
	sample_from_items = np.argsort(item_rating_counts)[-n_most:]

	probs_vs_true_groups_trial_users(sample_from_items=set(sample_from_items))

def probs_vs_true_groups_trial_users_least_ratings(n_least=1000):
	"""
	Sample items from the n most rated items only, to use with probs_vs_true_groups_trial_users.
	"""
	nym_stats = Data.get_nym_stats()
	item_rating_counts = nym_stats[:,:,3].astype(int).sum(axis=0)
	sample_from_items = np.argsort(item_rating_counts)[:n_least]

	probs_vs_true_groups_trial_users(sample_from_items=set(sample_from_items))

def probs_vs_true_groups_trial_users_low_var(n_lowest_var=100, f=np.min):
	"""
	Sample items with lowest variance
	f applied to variances of each group to determine overall variance of item 
	"""
	nym_stats = Data.get_nym_stats()
	variance_per_item = np.apply_along_axis(f, 0, nym_stats[:,:,2])
	sample_from_items = nym_stats[0, np.argsort(variance_per_item)[:n_lowest_var], 0].astype(int)

	probs_vs_true_groups_trial_users(sample_from_items=set(sample_from_items))

def probs_vs_true_groups_trial_users_low_var_thresh(n_lowest_var=100, f=np.min, thresh=25):
	"""
	Sample items with lowest variance
	f applied to variances of each group to determine overal variance of item
	"""
	nym_stats = Data.get_nym_stats()

	thresh_items = np.all(np.apply_along_axis(lambda x: x >= thresh, 0, nym_stats[:,:,3]), axis=0)

	total_item_count = Data.get_nym_stats().shape[1]
	thresh_item_count = thresh_items.sum()
	print(f'Using {thresh_item_count/total_item_count:#.3} of items ({thresh_item_count} of {total_item_count})')

	nym_stats = nym_stats[:,thresh_items,:]

	variance_per_item = np.apply_along_axis(f, 0, nym_stats[:,:,2])
	sample_from_items = nym_stats[0, np.argsort(variance_per_item)[:n_lowest_var], 0].astype(int)

	probs_vs_true_groups_trial_users(sample_from_items=set(sample_from_items))

def probs_vs_true_groups_trial_users_thresh(thresh=25, print_stats=True, **kwargs):
	"""
	Sample items with number ratings by groups above threshold
	"""
	nym_stats = Data.get_nym_stats()
	
	thresh_items = np.all(np.apply_along_axis(lambda x: x >= thresh, 0, nym_stats[:,:,3]), axis=0)
	total_item_count = Data.get_nym_stats().shape[1]
	thresh_item_count = thresh_items.sum()
	if print_stats:
		print(f'Using {thresh_item_count/total_item_count:#.3} of items ({thresh_item_count} of {total_item_count})')

	sample_from_items = nym_stats[0,thresh_items,0].astype(int)
	return probs_vs_true_groups_trial_users(sample_from_items=set(sample_from_items), print_stats=print_stats, **kwargs)

def plot_thresh_vs_mean_accuracy():

	# regs = [None, 5, 10, 25]
	regs = [6, 8, 10, 12]
	threshs = 50
	for reg_n, reg in enumerate(regs):
		means = np.zeros(threshs)
		x = np.arange(threshs) + 1
		for i, thresh in enumerate(x):
			correct_preds = probs_vs_true_groups_trial_users_thresh(thresh, users=100, samples=100, print_stats=False, dist_reg=reg)
			means[i] = correct_preds.mean()

		plt.subplot(2, 2, reg_n+1)
		plt.xlabel(f'threshold [reg = {reg}]')
		plt.ylabel('accuracy')
		plt.axhline(0.40, c='r', linestyle=':')
		plt.plot(x, means)
		plt.ylim(0.25, 0.55)
	
	plt.suptitle(f'Popularity threshold vs prediction accuracy')
	plt.show()

def plot_items_per_samples_vs_mean_accuracy():
	means = []
	max_per = 50
	x = np.arange(max_per) + 1
	for samples in x:
		correct_preds = probs_vs_true_groups_trial_users_thresh(samples=100, users=100, items_per_sample=samples, print_stats=False)
		means.append(correct_preds.mean())

	plt.title('Items per sample vs mean accuracy of group prediction')
	plt.xlabel('items per sample')
	plt.ylabel('mean accuracy')
	plt.plot(x, means)
	plt.show()


if __name__ == '__main__':
	# probs_vs_true_groups_real_data()
	# probs_vs_true_groups_trial_users()
	# probs_vs_true_groups_trial_users(users=1, items_per_sample=20)
	
	# probs_vs_true_groups_trial_users_most_ratings()
	# probs_vs_true_groups_trial_users_least_ratings(n_least=100)
	
	# probs_vs_true_groups_trial_users_low_var(f=np.min)
	# probs_vs_true_groups_trial_users_low_var(f=np.sum)
	# probs_vs_true_groups_trial_users_low_var(f=np.linalg.norm)
	# probs_vs_true_groups_trial_users_low_var_thresh()
	
	# probs_vs_true_groups_trial_users_thresh()
	
	# probs_vs_true_groups_trial_users_thresh(samples=1, users=10000, items_per_sample=100)

	plot_thresh_vs_mean_accuracy()
	# plot_items_per_samples_vs_mean_accuracy()


## Idea:
# For one user, check average accuracy over many trials, with some items_per_trial (e.g. 5)
# Then restrict the data by rating count threshold, etc.
# Then try use utility
##