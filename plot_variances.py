import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import argparse

from datareader import DataReader as Data
from myutils import msg

thresh_default = 50
outfile_default = Data.figure_dir + "variances.png"

parser = argparse.ArgumentParser(description="Plot variance of each group by item number. The size of each bubble corresponds to the square root of the number of ratings for that distribution. Only bubbles with at least the threshold number of ratings are plotted.")
parser.add_argument("-b", help="index of the item to begin plotting from", default=None, type=int)
parser.add_argument("-n", help="number of items to plot", default=None, type=int)
parser.add_argument("-t", help=f"only plot distributions with at least threshold number of ratings (defualt {thresh_default})", default=thresh_default, type=int)
parser.add_argument("-i", help="plot inverse variance instead of variance", action="store_true")
parser.add_argument("--savefig", help="save the figure to file rather than displaying the figure", action="store_true")
parser.add_argument("-outfile", help=f'file to save the figure to (default "{outfile_default}")', default=outfile_default)

def plot_variances(thresh=thresh_default, inv=False, savefig=False, outfile=outfile_default, begin=None, num=None):
	inv_msg = "inverse variance" if inv else "variance"
	
	fig, ax = plt.subplots()
	ax.set(
		# ylim=(0, None),
		title=f'variance of each group by item number (thresh no. ratings >= {thresh})',
		xlabel='item number',
		ylabel=inv_msg)
	
	cm = plt.get_cmap('gist_rainbow')
	colors = [cm(1.*i/Data.nym_count()) for i in range(Data.nym_count())]

	begin = 0 if begin is None else begin
	end = None if num is None else begin + num 
	nym_stats = Data.get_nym_stats()[:, begin : (None if num is None else begin+num),:]

	for nym_n in range(Data.nym_count()):
		nym_n_stats = nym_stats[nym_n]
		with msg(f'plotting nym #{nym_n} {inv_msg}'):

			valids = (nym_n_stats[:,3] >= thresh)
			print(f'{valids.sum()} of {len(valids)} valid (thresh = {thresh})')

			x = nym_n_stats[:,0][valids]
			y = nym_n_stats[:,2][valids]
			if inv: y[y > 0] = 1 / y[y > 0]
			s = np.sqrt(nym_n_stats[:,3][valids])

			ax.scatter(x, y, s=s, facecolors='none', edgecolors=colors[nym_n], label=f'group {nym_n}')
	ax.legend()

	if savefig:
		with msg('Saving "{}" to "{}"'.format(ax.title.get_text(), outfile)):
			ax.get_figure().savefig(outfile, dpi=150)
			plt.clf()
	else:
		plt.show()

if __name__ == "__main__":
	args = parser.parse_args()
	plot_variances(args.t, args.i, args.savefig, args.outfile, args.b, args.n)