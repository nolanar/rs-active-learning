from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

class DiscreteNormal:

	def __init__(self, bins):
	 	self.bins = bins

	def prob(self, bin, mean, var):
		""" probability of specific bin """
		if bin <= 0:
			return norm.cdf(self.bins[1], mean, np.sqrt(var))
		if bin >= len(self.bins) - 2:
			return 1 - norm.cdf(self.bins[bin], mean, np.sqrt(var))

		start, end = norm.cdf((self.bins[bin], self.bins[bin+1]), mean, np.sqrt(var))
		return end - start

	def pmf(self, mean, var):
		""" probability mass function of bins """
		bins = self.bins[:-1]
		starts = norm.cdf(bins, mean, np.sqrt(var))
		ends = np.roll(starts, -1)
		ends[-1] = 1
		starts[0] = 0
		return ends - starts

def example_pmf_plot(mean, var):
	bins = np.linspace(0.5, 5.5, num=6)
	dnorm = DiscreteNormal(bins)

	disc_y = dnorm.pmf(mean, var)
	plt.bar(range(1, 6), disc_y, color='lightgrey', edgecolor='black')

	xlim = (0.5,5.5)
	x = np.linspace(xlim[0], xlim[1], num=100)
	plt.plot(x, norm.pdf(x, mean, np.sqrt(var)), color='black', linewidth=2)
	plt.xlim(xlim)

	plt.title(f'Discrete approx. of normal distribution (mean = {mean} var = {var})')
	plt.show()

def example_prob_print(mean, var):
	bins = np.linspace(0.5, 5.5, num=6)
	dnorm = DiscreteNormal(bins)
	for i in range(5):
		print(f'bin {i + 1} prob:', dnorm.prob(i, mean, var))

def rv_test(var):
	# print(f'ideal: mean = {mean} var = {var}')

	bins = np.linspace(0.5, 5.5, num=6)
	dnorm = DiscreteNormal(bins)

	stats = np.zeros((100, 2), dtype=np.float32)
	x = np.linspace(1, 5, num=100)
	for i, mean in enumerate(x):
		ps = dnorm.pmf(mean, var)
		samples = np.random.choice(np.arange(1,6), size=100000, p=ps)
		stats[i, 0] = samples.mean()
		stats[i, 1] = samples.var()
	
	# plt.plot(x, x, label='true mean', ls='dashed', c='b')	
	# plt.plot(x, abs(x - stats[:,0]), label='error', ls='dotted', c='b')	
	# plt.plot(x, stats[:,0], label='sample mean', c='b')
	
	plt.axhline(var, label='true var', c='g', ls='dashed')
	plt.plot(x, stats[:,1], label='sample var', c='g')
	
	plt.legend()
	plt.show()

if __name__ == '__main__':
	mean, var = 4.5, 2
	example_prob_print(mean, var)
	# example_pmf_plot(mean, var)
	rv_test(var)