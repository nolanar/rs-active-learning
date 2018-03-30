import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import argparse

from datareader import DataReader as Data
from myutils import msg

thresh_default = 50
outfile_default = Data.figure_dir + "variances.png"

parser = argparse.ArgumentParser(description="plot vairance of each group by item number")
parser.add_argument("--thresh", help=f"only plot distributions with at least thresh number of ratings (defualt {thresh_default})", default=thresh_default, type=int)
parser.add_argument("--inverse", help="plot inverse variance instead of variance", action="store_true")
parser.add_argument("--savefig", help="save the figure to file rather than displaying the figure", action="store_true")
parser.add_argument("--outfile", help=f'file to save the figure to (default "{outfile_default}")', default=outfile_default)
parser.add_argument("--rmin", help="Lowest index of item to display", default=None, type=int)
parser.add_argument("--rmax", help="Highest index of item to display (inclusive)", default=None, type=int)

def plot_variances(thresh=thresh_default, inv=False, savefig=False, outfile=outfile_default, rmin=None, rmax=None):
	inv_msg = "inverse variance" if inv else "variance"
	
	fig, ax = plt.subplots()
	ax.set(
		# ylim=(0, None),
		title=f'vairance of each group by item number (thresh no. ratings >= {thresh})',
		xlabel='item number',
		ylabel=inv_msg)
	
	cm = plt.get_cmap('gist_rainbow')
	colors = [cm(1.*i/Data.nym_count) for i in range(Data.nym_count)]

	nym_stats = Data.get_nym_stats()[:, rmin : (None if rmax is None else rmax+1),:]

	for nym_n in range(Data.nym_count):
		nym_n_stats = nym_stats[nym_n]
		with msg(f'plotting {inv_msg}'):

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
	plot_variances(args.thresh, args.inverse, args.savefig, args.outfile, args.rmin, args.rmax)