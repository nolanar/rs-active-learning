import argparse
import os
from plot_item_dist import barplot_rating_dist
from datareader import DataReader as Data

parser = argparse.ArgumentParser(description="Plot distributions of each group for an item.")
parser.add_argument('item', help='item number to plot')
args = parser.parse_args()

item = args.item
fig_dir = f'figures/item{item}/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

for nym_n in range(Data.nym_count()):
	barplot_rating_dist(item, group=nym_n, savefig=f'{fig_dir}item{item}_group{nym_n}_dist.png')