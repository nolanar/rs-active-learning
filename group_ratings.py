import numpy as np
import random
from functools import lru_cache

from datareader import DataReader as Data
from myutils import msg

class GroupRatings:
    """
    Class to represent ratings on a set of items by groups.

    n = number of users
    m = number of items
    p = number of groups
    r = number of rating values
    """

    def __init__(self, group_ratings=None, groups=None, group_sizes=None, items=None, rating_vals=None, output=True):

        if group_ratings is not None: self.ratings = group_ratings
        else: self.ratings = Data.get_group_ratings()

        if rating_vals is not None: np.array(rating_vals)
        else: self.rating_vals = np.arange(Data.rating_value_count) + 1 # from 1 to rating_value_count
        self.rating_val_count = self.rating_vals.shape[0]

        # ID of items, for use with filtering
        if items is not None: self.items = items
        else: self.items = np.arange(self.ratings.shape[1])

        # ID of groups
        if groups is not None: self.groups = groups
        else: self.groups = np.arange(self.ratings.shape[0])

        if group_sizes is not None: self.g_sizes = group_sizes
        else: self.g_sizes = Data.group_sizes()

        self.default_reg = None   # regularisation to be used when not specified as arg in dist().
        self.output = output      # print execution information

    def get_ratings(self):
        return self.ratings[:, self.items, :]

    def get_items(self):
        """ IDs of items """
        return self.items

    def group_sizes(self):
        """ the number of users in each group """
        return self.g_sizes

    def group_size_dist(self):
        """ the (normalised) distribution of users in each group """
        return self.group_sizes() / self.group_sizes().sum()

    def dist(self, reg=None):
        """ normalised distribution of ratings by each group. [p x m x r] array """
        dist = self.get_ratings().copy()

        # Add small value to regularise low popularity items
        if reg is None: reg = self.default_reg
        if reg: dist += reg/dist.shape[2]
        
        # if a group has no ratings for an item, give it a uniform distribution
        sum_ratings = dist.sum(axis=2)
        zero_sum_ratings = sum_ratings == 0
        dist[zero_sum_ratings, :] = 1 / dist.shape[2]
        sum_ratings[zero_sum_ratings] = 1

        dist /= sum_ratings[:,:,None] # normalise
        return dist

    def lam(self):
        """ number of ratings by each group, on each item. [p x m] array """
        return self.get_ratings().sum(axis=2)

    def n_per_item(self):
        """ total number of ratings on each item. [m] array """
        return self.lam().sum(axis=0)

    def filter(self, items):
        """ Keep only the given items """
        self.items = np.intersect1d(self.items, self.items[items])

    def item_count(self):
        return self.items.shape[0]

    def means(self):
        return self.dist().dot(self.rating_vals)

    def vars(self):
        return self.dist().dot(self.rating_vals**2) - self.means()**2

    def stds(self):
        return np.sqrt(self.vars())

    def reg(self, reg):
        self.default_reg = reg
        with msg(f'Regularising with reg = {reg}', done=False, enabled=self.output):pass

    def thresh(self, thresh=25, total_ratings=False):
        """
        Filter out items which have groups below the thresh number of ratings.
        With total_item_ratings = True, the total number of ratings on each item is compared instead.
        """
        before = self.item_count()

        if total_ratings: self.filter(self.n_per_item() >= thresh)
        else: self.filter(np.all(self.lam() >= thresh, axis=0))

        after = self.item_count()
        with msg(f'Applying threshold of {thresh} : {after} of {before}', done=False, enabled=self.output):pass

    def highest_pop(self, n=100):
        """ Filter the n items with highest popularity """
        before = self.item_count()

        self.filter(np.argsort(self.n_per_item())[-n:])

        after = self.item_count()
        with msg(f'Using {n} with highest popularity: {after} of {before}', done=False, enabled=self.output):pass

    def lowest_pop(self, n=100):
        """ Filter the n items with lowest popularity """
        before = self.item_count()

        self.filter(np.argsort(self.n_per_item())[:n])

        after = self.item_count()
        with msg(f'Using {n} with lowest popularity: {after} of {before}', done=False, enabled=self.output):pass
 
    def lowest_var(self, n=100):
        """ Filter the n items with lowest variance """
        before = self.item_count()

        stds = np.amin(self.stds(), axis=0)
        self.filter(np.argsort(stds)[:n])

        after = self.item_count()
        with msg(f'Using {n} with lowest variance: {after} of {before}', done=False, enabled=self.output):pass

    def keep_n(self, n=100):
        """ Filter n items at random """
        before = self.item_count()

        item_count = self.item_count()
        if item_count > n: self.filter(self.sample(n))

        after = self.item_count()
        with msg(f'Keeping (at most) {n} items: {after} of {before}', done=False, enabled=self.output):pass


    def sample(self, n, items_per=None):
        if items_per is None:
            item_count = self.item_count()        
            return np.array(random.sample(set(range(item_count)), n))
        else:
            item_pool = self.items
            samples = np.array([self.sample(items_per) for _ in range(n)])
            samples.sort(axis=1)
            return samples
