import numpy as np
import random
from functools import lru_cache

import dist_uncertainty
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

    def __init__(self, group_ratings=None, Rtilde=None, groups=None, group_sizes=None, items=None, rating_vals=None, correct_error=True, output=True):

        if group_ratings is not None: self.ratings = group_ratings
        else: self.ratings = Data.get_group_ratings()

        if Rtilde is not None: self.Rtilde = Rtilde
        else: self.Rtilde = Data.get_Rtilde()

        if rating_vals is not None: np.array(rating_vals)
        else: self.rating_vals = np.arange(Data.rating_value_count) + 1 # from 1 to rating_value_count
        self.n_rating_vals = self.rating_vals.shape[0]

        # ID of items, for use with filtering
        if items is not None: self.items = items
        else: self.items = np.arange(self.ratings.shape[1])

        # ID of groups
        if groups is not None: self.groups = groups
        else: self.groups = np.arange(self.ratings.shape[0])
        self.n_groups = self.groups.shape[0]

        if group_sizes is not None: self.g_sizes = group_sizes
        else: self.g_sizes = Data.group_sizes()

        self.default_reg = None             # regularisation to be used when not specified as arg in dist().
        self.correct_error = correct_error  # use error correction
        self.output = output                # print execution information

    def get_ratings(self):
        return self.ratings[:, self.items, :]

    def get_Rtilde(self):
        return self.Rtilde[:, self.items]

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

    def dist(self, correct_error=None, reg=None):
        """ normalised distribution of ratings by each group. [p x m x r] array """
        if correct_error is None: correct_error = self.correct_error
        if correct_error:
            return dist_uncertainty.get_uncertainty_adjusted_dist(self)[:, self.items, :]

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

    def reset(self):
        """ reset the list of filtered items to all items """
        self.items = np.arange(self.ratings.shape[1])

    def item_count(self):
        return self.items.shape[0]

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
        Get the rmse of the entire dataset.
        """
        lam = self.lam()
        weights = lam / lam.sum()
        weighted_var = self.var() * weights
        rmse = np.sqrt(weighted_var.sum())
        return rmse

    def reg(self, reg):
        self.default_reg = reg
        with msg(f'Regularising with reg = {reg}', done=False, enabled=self.output):pass

    def thresh(self, thresh=25, total_ratings=False):
        """
        Filter out items which have groups below the thresh number of ratings.
        With total_ratings = True, the total number of ratings on each item is compared instead.
        """
        before = self.item_count()

        if total_ratings: self.filter(self.n_per_item() >= thresh)
        else: self.filter(np.all(self.lam() >= thresh, axis=0))

        after = self.item_count()
        thresh_type = 'on each item total'  if total_ratings else 'by each group' 
        with msg(f'Applying threshold of {thresh} ratings {thresh_type} : {after} of {before}', done=False, enabled=self.output):pass

    def highest_pop(self, n=100):
        """ Filter the n items with highest popularity """
        self.highest_x(n, self.n_per_item(), 'popularity')

    def lowest_pop(self, n=100):
        """ Filter the n items with lowest popularity """
        self.lowest_x(n, self.n_per_item(), 'popularity')

    def highest_var(self, n=100):
        """ Filter the n items with highest variance """
        self.highest_x(n, np.amin(self.sd(), axis=0), 'variance')

    def lowest_var(self, n=100):
        """ Filter the n items with lowest variance """
        self.lowest_x(n, np.amin(self.sd(), axis=0), 'variance')

    def lowest_entropy(self, n, priors=None):
        """ Filter the n items with lowest conditional entropy """
        self.lowest_x(n, self.entropy(priors), 'entropy')

    def highest_maxnorm(self, n, priors=None):
        """ Filter the n items with highest maxnorm """
        self.highest_x(n, self.maxnorm(priors), 'maxnorm')

    def highest_pnorm(self, n, p=2, priors=None):
        """ Filter the n items with highest p-norm """
        self.highest_x(n, self.pnorm(p, priors, False), f'{p}-norm')

    def highest_cocondition(self, n, priors=None):
        conconds = self.cocondition(priors).sum(axis=0)
        self.highest_x(n, conconds, f'cocondition')

    def lowest_x(self, n, x, description):
        """ Filter the n items with lowest x """
        before = self.item_count()
        self.filter(np.argsort(x)[:n])
        after = self.item_count()
        with msg(f'Using {n} with lowest {description}: {after} of {before}', done=False, enabled=self.output):pass

    def highest_x(self, n, x, description):
        """ Filter the n items with highest x """
        before = self.item_count()
        self.filter(np.argsort(x)[-n:])
        after = self.item_count()
        with msg(f'Using {n} with highest {description}: {after} of {before}', done=False, enabled=self.output):pass

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
            return self.fast_sample(n, items_per)
            # item_pool = self.items
            # samples = np.array([self.sample(items_per) for _ in range(n)])
            # samples.sort(axis=1)
            # return samples

    def fast_sample(self, n, items_per=None):
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

    def entropy(self, priors=None):
        def entropy_f(x):
            x[x != 0] *= np.log(x[x != 0])
            return -x.sum(axis=0)
        return self.utility(entropy_f, priors)

    def maxnorm(self, priors=None):
        def maxnorm_f(x): return x.max(axis=0)
        return self.utility(maxnorm_f, priors)

    def pnorm(self, p, priors=None, root=True):
        if root: pnorm_f = lambda x: (x**p).sum(axis=0)**(1/p)
        else: pnorm_f = lambda x: (x**p).sum(axis=0)
        return self.utility(pnorm_f, priors)

    def posteriors(self, priors=None, return_r=False):
        if priors is None: priors = self.group_size_dist()
        posteriors = self.dist() * priors.T[None,None].T    # (n)x p x m x r
        prob_r = posteriors.T.sum(axis=2, keepdims=True).T  # (n)x 1 x m x r
        posteriors /= prob_r                                # (n)x p x m x r
        if return_r: return posteriors, prob_r
        else: return posteriors

    def utility(self, util_f, priors=None):
        if priors is None: priors = self.group_size_dist()
        posteriors = self.dist() * priors[:,None,None] # p x m x r
        prob_R = posteriors.sum(axis=0)                #     m x r
        posteriors /= prob_R                           # p x m x r
        return (util_f(posteriors) * prob_R).sum(axis=1)

    def cocondition(self, priors=None):
        posts = self.posteriors(priors)               # (n)x p x m x r
        return (posts * self.dist()).T.sum(axis=0).T  # (n)x p x m

    def max_cocondition(self, priors=None):
        return np.argmax(self.cocondition(priors).T.sum(axis=1), axis=0).T

if __name__ == '__main__':
    g = GroupRatings(correct_error=False)
    # priors = np.ones(8)
    # # priors = np.arange(16).reshape(2, 8)
    # priors = priors / priors.sum(axis=0)

    # print(g.max_cocondition(priors))

    g.thresh(100)
    # g.lowest_var(100)
    print(g.rmse())