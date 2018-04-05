import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import seaborn as sns 
import argparse

from datareader import DataReader as Data
from myutils import msg

parser = argparse.ArgumentParser(description="Plot of the group distributions for an item.")
parser.add_argument('item', help='number of item to plot')
parser.add_argument('-g', help='group number to plot (all by default)', default=None, type=int)
parser.add_argument('-a', help='plot all ratings together ignoring groups', action="store_true")
parser.add_argument('--heat', help='plot as 2D heatmap (defualt histogram)', action="store_true")

def distplot(d, **kwargs):
	step = 1
	bins = np.arange(step/2, 5 + 1.5*step, step)
	sns.distplot(d, bins=bins, kde=False, hist_kws={"linewidth": 0, "alpha": 1}, **kwargs)

def barplot_rating_dist(item, kde=False, single=False, group=None):

	with msg("plotting rating distribution"):
		ratings = Data.get_ratings()[:,item]
		nyms = Data.get_nyms()

		if group is not None: 
			distplot(ratings[nyms[group]].data, label=f'group {group}')
		elif single: 
			distplot(ratings.data, label=f'all item #{item} ratings')
		else:
			for nym_n, nym in enumerate(nyms):
				distplot(ratings[nym].data, label=f'group {nym_n}')
		plt.legend()
		plt.show()

def heatmap_rating_dist(item):
	# def plot_rating_dists_across_groups(ratings, item, groups, savefig=False):
	with msg("plotting rating distribution"):
		ratings = Data.get_ratings()[:,item]
		nyms = Data.get_nyms()

		data = np.zeros((10, len(nyms)))
		for nym_n, nym in enumerate(nyms):
			unique, count = np.unique(ratings[nym].data, return_counts=True)
			for rating, count in dict(zip(unique, count)).items():
				data[int(2*rating - 1), nym_n] = count

		ax = sns.heatmap(data)
		ax.set(
			title="Distribution of item #{} ratings by group".format(int(item)),
			xlabel="group number", 
			ylabel="rating", 
			yticklabels=np.linspace(0.5, 5, 10))
		
		plt.show()

if __name__ == "__main__":
	args = parser.parse_args()
	if (args.heat): heatmap_rating_dist(args.item)
	else: barplot_rating_dist(args.item, single=args.a, group=args.g)
