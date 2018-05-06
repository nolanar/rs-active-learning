import numpy as np
import random
from functools import lru_cache
from myutils import *


class GroupRatings:
    """
    Class to represent group ratings on items.
    """

    def __init__(self, output=True):
        self.output = output  # print running information
        import data
        self.ratings = data.get_group_ratings()
        self.rating_vals = np.arange(data.rating_value_count) + 1 # from 1 to rating_value_count
        self.n_rating_vals = self.rating_vals.shape[0]
        self.items = np.arange(self.ratings.shape[1]) # ID of items, for use with filtering
        self.groups = np.arange(self.ratings.shape[0]) # ID of groups
        self.n_groups = self.groups.shape[0]
        self.g_sizes = data.group_sizes()

    def get_ratings(self):
        return self.ratings[:, self.items, :]

    def item_ratings(self):
        """ the number of ratings each item has in each rating value """
        return self.get_ratings().sum(axis=0)

    def get_items(self):
        """ IDs of items """
        return self.items

    def group_sizes(self):
        """ the number of users in each group """
        return self.g_sizes

    def group_size_dist(self):
        """ the (normalised) distribution of users in each group """
        return self.group_sizes() / self.group_sizes().sum()

    def lam(self):
        """ number of ratings by each group, on each item. [p x m] array """
        return self.get_ratings().sum(axis=2)

    def n_per_item(self):
        """ total number of ratings on each item. [m] array """
        return self.lam().sum(axis=0)

    def item_count(self):
        """ Return the number of items """
        return self.items.shape[0]

    #### Distributions and statistics ####

    def dist(self):
        """ normalised distribution of ratings by each group. [p x m x r] array """
        dist = self.get_ratings()
        sum_ratings = dist.sum(axis=2, keepdims=True)
        with suppress_numpy_err(): # suppress warning about nan values
            dist = dist / sum_ratings # normalise
        dist[np.isnan(dist)] = 1 / dist.shape[0] # if a group has no ratings for an item, give it a uniform distribution
        return dist

    def mean(self, dist=None):
        """ get mean of each distribution """
        if dist is None: dist = self.dist()
        return dist.dot(self.rating_vals)

    def var(self, dist=None):
        """ get variance of each distribution """
        if dist is None: dist = self.dist()
        return dist.dot(self.rating_vals**2) - self.mean(dist)**2

    def sd(self, dist=None):
        """ get standard deviation of each distribution """
        return np.sqrt(self.var(dist))

    def rmse(self):
        """ 
        Get the expected rmse of the entire dataset.
        """
        lam = self.lam()
        weights = lam / lam.sum()
        weighted_var = self.var() * weights
        rmse = np.sqrt(weighted_var.sum())
        return rmse

    #### Item filtering ####

    def filter(self, items, relative=True):
        """ Keep only the given items """
        if relative: items = self.items[items]
        self.items = np.intersect1d(self.items, items)

    def remove(self, items, relative=True):
        """ Remove the given items """
        if relative: items = self.items[items]
        self.items = np.setdiff1d(self.items, items)

    def reset(self):
        """ Reset the list of filtered items to all items """
        self.items = np.arange(self.ratings.shape[1])

    def thresh(self, thresh=25, total_ratings=False):
        """
        Remove items which are below the thresh number of ratings.
        total_ratings = False    Must have each group above thresh number of ratings
        total_ratings = True     Total number of ratings acorss all groups must be above thresh
        """
        before = self.item_count()

        if total_ratings: self.filter(self.n_per_item() >= thresh)
        else: self.filter(np.all(self.lam() >= thresh, axis=0))

        after = self.item_count()
        thresh_type = 'on each item total'  if total_ratings else 'by each group' 
        with msg(f'Applying threshold of {thresh} ratings {thresh_type} : {after} of {before}', done=False, enabled=self.output):pass

    def highest_pop(self, n=100):
        """ Keep the n items with highest popularity """
        self.highest_x(n, self.n_per_item(), 'popularity')

    def lowest_pop(self, n=100):
        """ Keep the n items with lowest popularity """
        self.lowest_x(n, self.n_per_item(), 'popularity')

    def highest_var(self, n=100):
        """ Keep the n items with highest variance """
        self.highest_x(n, np.amin(self.sd(), axis=0), 'variance')

    def lowest_var(self, n=100):
        """ Keep the n items with lowest variance """
        self.lowest_x(n, np.amin(self.sd(), axis=0), 'variance')

    def lowest_entropy(self, n, priors=None):
        """ Keep the n items with lowest conditional entropy """
        self.lowest_x(n, self.entropy(priors), 'entropy')

    def highest_maxnorm(self, n, priors=None):
        """ Keep the n items with highest maxnorm """
        self.highest_x(n, self.maxnorm(priors), 'maxnorm')

    def highest_pnorm(self, n, p=2, priors=None):
        """ Keep the n items with highest p-norm """
        self.highest_x(n, self.pnorm(p, priors, False), f'{p}-norm')

    def lowest_x(self, n, x, description):
        """ Keep the n items with lowest x """
        before = self.item_count()
        self.filter(np.argsort(x)[:n])
        after = self.item_count()
        with msg(f'Using {n} with lowest {description}: {after} of {before}', done=False, enabled=self.output):pass

    def highest_x(self, n, x, description):
        """ Keep the n items with highest x """
        before = self.item_count()
        self.filter(np.argsort(x)[-n:])
        after = self.item_count()
        with msg(f'Using {n} with highest {description}: {after} of {before}', done=False, enabled=self.output):pass

    def keep_n(self, n=100):
        """ Keep n items at random """
        before = self.item_count()

        item_count = self.item_count()
        if item_count > n: self.filter(self.sample(n))

        after = self.item_count()
        with msg(f'Keeping (at most) {n} items: {after} of {before}', done=False, enabled=self.output):pass

    #### Measures of utility ####

    def entropy(self, priors=None):
        """ Return entropy of each item """
        def entropy_f(x):
            x[x != 0] *= np.log(x[x != 0])
            return -x.sum(axis=0)
        return self.utility(entropy_f, priors)

    def maxnorm(self, priors=None):
        """ Return max-norm of each item """
        def maxnorm_f(x): return x.max(axis=0)
        return self.utility(maxnorm_f, priors)

    def pnorm(self, p, priors=None, root=True):
        """ Return p-norm of each item """
        if root: pnorm_f = lambda x: (x**p).sum(axis=0)**(1/p)
        else: pnorm_f = lambda x: (x**p).sum(axis=0)
        return self.utility(pnorm_f, priors)

    def utility(self, util_f, priors=None):
        """ Base for other utility functions """
        if priors is None: priors = self.group_size_dist()
        posteriors = self.dist() * priors[:,None,None] # p x m x r
        prob_R = posteriors.sum(axis=0)                #     m x r
        posteriors /= prob_R                           # p x m x r
        return (util_f(posteriors) * prob_R).sum(axis=1)

    #### Sampling ####

    def sample(self, n, items_per=1, weight=False):
        """
        Sample n item sets, with "items_per" items per sample.
        weight = False   Sample items with uniform weighting (using fast_sample)
        weight = True    Sample according to item popularity
        """
        if weight:
            item_count = self.item_count()
            p = self.n_per_item()
            p = p / p.sum()
            return np.array([np.random.choice(item_count, size=items_per, replace=False, p=p) for _ in range(n)])
        else:
            return self.fast_sample(n, items_per)

    def fast_sample(self, n, items_per=None):
        """ Sample by shuffling items, then chopping into sample sets, then repeating until n sample sets produced """
        item_pool = np.arange(self.items.shape[0]) #self.items.copy()
        samples = []
        remaining = n
        samples_per_shuffle = int(item_pool.shape[0]/items_per)
        while remaining > 0:
            random.shuffle(item_pool)
            for i in range(0, min(samples_per_shuffle, remaining) * items_per, items_per):
                samples.append(item_pool[i:i+items_per])
                remaining -= 1
        return np.array(samples)

if __name__ == '__main__':
    g = GroupRatings()
    g.thresh(60)

    # g.lowest_var(100)
    # print(g.rmse())
    # with msg('Sampling'): print(g.sample(1000, items_per=25, weight=False))

