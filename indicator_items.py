import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt

from myutils import msg
from datareader import DataReader as Data
from user import synthesise_user_data
from items import Items
from group_ratings import GroupRatings

class IndicatorItemsFinder:

    def __init__(self, group_ratings, sample_n=1000, items_per_sample=5, user_n=1000, group_ratios=None, priors=None, output=True):

        self.sample_n = sample_n                  # number of samples
        self.user_n = user_n                      # number of users
        self.items_per_sample = items_per_sample  # number of items in each sample
        self.output = output                      # print execution information

        self.group_ratings = group_ratings  

        # prior group probabilities
        if priors is not None: self.priors = priors
        else: self.priors = self.group_ratings.group_size_dist() # use distribution of true group sizes by default

        # ratio of users in each group
        if group_ratios is not None: self.group_ratios = group_ratios
        else: self.group_ratios = self.group_ratings.group_sizes() # use sizes of true groups by default

        # Set after calling run():
        self.has_been_run   = False
        self.samples        = None
        self.unique_samples = None
        self.user_ratings   = None
        self.user_groups    = None
        self.posteriors     = None

    def run(self):
        with msg(f'Getting {self.sample_n} samples, {self.items_per_sample} items per', enabled=self.output):
            self.samples = self.group_ratings.sample(self.sample_n, self.items_per_sample)
            self.unique_samples = np.unique(self.samples)

        with msg(f'Synthesising {self.user_n} users, each with {self.unique_samples.shape[0]} item ratings', enabled=self.output):
            self.user_ratings, self.user_groups = synthesise_user_data(
                self.group_ratings.dist(), self.user_n, self.group_ratios, self.group_ratings.rating_val_count)

        with msg('Calculating posterior probabilities', enabled=self.output):
            self.posteriors = self.get_group_probs(self.group_ratings.dist(), self.user_ratings, self.samples, self.priors)

    def get_accuracy(self, probs=None, user_groups=None, user_accuracy=False):
        if probs is None:
            if self.posteriors is None: self.run()
            probs = self.posteriors
        if user_groups is None: 
            if self.user_groups is None: self.run()
            user_groups = self.user_groups

        mean_axis = 0 if user_accuracy else 1
        return (probs.argmax(axis=2) == user_groups).mean(axis=mean_axis)

    def get_group_probs(self, group_ratings_dist, user_ratings, samples, priors):
        """
        Get the postirior probabilities of group membership after each sample has been 
        rated by each user, using Bayes' theorem.

        P(R=r|G_u)'s (from group_ratings_dist) are assumed to be independent.
        
        Array               Dimensions   Values            Key
        user_ratings        = n x m      = r               n = number of users
        samples             = k x q      = m               m = number of items
        priors              = n x p                        p = number of groups
        group_ratings_dist  = p x m x r                    r = number of (unique) ratings
                                                           k = number of samples
                                                           q = number of items per sample
        """
        probs = group_ratings_dist[:, samples, user_ratings[:,samples]].prod(axis=3).T # k x n x p 
        probs *= priors # k x n x p * p  (or  k x n x p * n x p)
        probs /= probs.sum(axis=2, keepdims=True) # normalise
        return probs # k x n x p

def print_stats():

    with msg('\nConfiguring group ratings selection'):
        group_ratings = GroupRatings()
        group_ratings.reg(8)
        # group_ratings.thresh(25)
        group_ratings.lowest_pop(1000)
        # group_ratings.highest_pop(200)
        # group_ratings.lowest_var(30)
        # group_ratings.keep_n(10)
    print()

    finder = IndicatorItemsFinder(group_ratings)
    accuracy = finder.get_accuracy()

    print(f'\n{finder.user_n} users, {finder.sample_n} samples, {finder.items_per_sample} items per sample:')
    print(f'min:\t{accuracy.min():#.3}')
    print(f'max:\t{accuracy.max():#.3}')
    print(f'mean:\t{accuracy.mean():#.3}')
    print(f'median:\t{np.median(accuracy):#.3}') 
    print()   

def plot_thresh_vs_accuracy_varying_reg():
    plt.title("Threshold vs accuracy \w 100 lowest pop")
    plt.xlabel('threshold')
    plt.ylabel('accuracy')

    for reg in [0, 1, 3, 5, 8, 13, 21, 34]: 
        count, step = 50, 2
        mean_accs = np.zeros(count)

        for i, thresh in enumerate(range(0, count*step, step)):
            group_ratings = GroupRatings()
            group_ratings.reg(reg)
            group_ratings.thresh(thresh)
            group_ratings.lowest_pop(100)
            # group_ratings.highest_pop(10)
            # group_ratings.lowest_var(100)
            group_ratings.keep_n(1000)

            mean_accs[i] = IndicatorItemsFinder(item_pool).get_accuracy()

        plt.plot(np.arange(count)*step, mean_accs, label=f'reg = {reg}')

    plt.legend()
    plt.show()

print_stats()