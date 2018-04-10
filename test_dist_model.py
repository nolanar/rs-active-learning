import numpy as np
import matplotlib.pyplot as plt

from myutils import msg
from datareader import DataReader
from dist_model import DiscreteNormal as DiscNorm

rating_count = 5
dist_gen = DiscNorm(np.linspace(0.5, 5.5, num=rating_count+1))

with msg("Getting data"):
	Rtilde = DataReader.get_Rtilde()
	Rvar = DataReader.get_Rvar()
	R = DataReader.get_ratings()
	lam = DataReader.get_lam()
	P = DataReader.get_nyms()

def get_data_dist(data):
	ratings, counts = np.unique(data, return_counts=True)
	dist_data = np.zeros(rating_count)
	dist_data[ratings.astype(int) - 1] = counts / counts.sum()
	return dist_data

def get_err(data, mean, var):
	dist_data = get_data_dist(data)
	dist_model = dist_gen.pmf(mean, var)
	return abs(dist_data / dist_model)

def get_rmse(data, mean, var):
	return np.sqrt(np.sum(get_err(data, mean, var) ** 2))

def plot_data_vs_model(data, mean, var, title=None, savefig=None):
	dist_data = get_data_dist(data)
	dist_model = dist_gen.pmf(mean, var)
	x = np.arange(1,6)
	w = 0.75 / 2
	plt.bar(x - w/2, dist_data, width=w*0.85, label='data')
	plt.bar(x + w/2, dist_model, width=w*0.85, label='model')
	if title: plt.title(title)
	plt.legend()
	
	if savefig:
		plt.savefig(savefig)
		plt.clr()
	else:
		plt.show()

def plot_highest_pop_items(group=0, highest_n=5):
	group_data = R[P[group]]

	item_lam = lam.sum(axis=0)
	highest_n = 5
	large_items = np.argpartition(item_lam, -highest_n)[-highest_n:]

	for item in large_items:
		data = group_data[:,item].data
		plot_data_vs_model(data, Rtilde[group, item], Rvar[group, item], 
			title=f'Item {item} ({lam[group, item]} total group {group} ratings)',
			savefig=f'figures/data_vs_model')

def total_rmse():
	group_count = DataReader.nym_count()
	item_count = R.shape[1]
	total_rmse = 0

	item_lam = lam.sum(axis=0)
	highest_n = 500
	large_items = np.argpartition(item_lam, -highest_n)[-highest_n:]

	with msg('Splitting group ratings'):
		group_ratings = []
		for group in range(group_count):
			group_ratings.append(R[P[group]])

	with msg('Getting rmse(s)'):
		count = 0
		for nth_item, item in enumerate(large_items):
			for group in range(group_count):
				mean = Rtilde[group, item]
				# if mean < 3.5 and mean > 2.5:
				# if mean > 4:
				if True:
					count += 1
					data = group_ratings[group][:,item].data
					var = Rvar[group, item]
					if var == 0: var = 0.01
					total_rmse += get_rmse(data, mean, var)
			
			if (nth_item) % 10 == 0:
				mean_rmse = total_rmse/(count)
				print(f'[{nth_item}, {count}] Mean RMSE: {mean_rmse}')

def total_err():
	group_count = DataReader.nym_count()
	item_count = R.shape[1]
	total_errs = np.zeros(rating_count)

	item_lam = lam.sum(axis=0)
	highest_n = 500
	large_items = np.argpartition(item_lam, -highest_n)[-highest_n:]

	with msg('Splitting group ratings'):
		group_ratings = []
		for group in range(group_count):
			group_ratings.append(R[P[group]])

	with msg('Getting errs'):
		count = 0
		for nth_item, item in enumerate(large_items):
			for group in range(group_count):
				mean = Rtilde[group, item]
				# if mean < 3.5 and mean > 2.5:
				# if mean > 4:
				if True:
					count += 1
					data = group_ratings[group][:,item].data
					var = Rvar[group, item]
					if var == 0: var = 0.01
					total_errs += get_err(data, mean, var)
			
			if (nth_item) % 10 == 0:
				mean_errs = total_errs/(count)
				print(f'[{nth_item}, {count}] Mean errs: {mean_errs}')

total_err()


# NOTES: 
# all data rmse ~0.13 
# 500 largest: ~0.08
# 500 largest, mean > 4: ~0.13
# 500 largest, mean in (2.5, 3.5):  ~0.06