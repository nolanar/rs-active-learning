import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import seaborn as sns 
import argparse

from datareader import DataReader as Data
from myutils import msg

parser = argparse.ArgumentParser(description="Plot of the group distributions for an item. Bar plots are used by default.")
parser.add_argument('item', help='number of item to plot')
parser.add_argument('--heat', help='plot as 2D heatmap', action="store_true")
parser.add_argument('--kde', help='plot as kernel density estimates', action="store_true")

def barplot_rating_dist(item, kde=False):
	with msg("plotting rating distribution"):
		ratings = Data.get_ratings()[:,item]
		nyms = Data.get_nyms()

		step = 0.5

		bins = np.arange(step, 5 + 2*step, step)
		for nym in nyms:
			data = (ratings[nym]).data + step/2
			if kde: sns.kdeplot(data, bw=0.5)
			else : sns.distplot(data, bins=bins, kde=False, hist_kws={"linewidth": 0, "alpha": 1})
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
	else: barplot_rating_dist(args.item, args.kde)
