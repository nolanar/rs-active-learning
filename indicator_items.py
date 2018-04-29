import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
import seaborn as sns

from myutils import msg
from datareader import DataReader as Data
from user import synthesise_user_data
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
        self.has_not_been_run = True
        self.samples          = None
        self.user_ratings     = None
        self.items            = None
        self.user_groups      = None
        self.posteriors       = None

    def run(self):
        with msg(f'Synthesising {self.user_n} users, each with {self.group_ratings.dist().shape[1]} item ratings', enabled=self.output):
            self.user_ratings, self.user_groups = synthesise_user_data(
                self.group_ratings.dist(), self.user_n, self.group_ratios, self.group_ratings.n_rating_vals)
            self.items = self.group_ratings.items

        if self.items_per_sample > 0:
            with msg(f'Getting {self.sample_n} samples, {self.items_per_sample} items per', enabled=self.output):
                self.samples = self.group_ratings.sample(self.sample_n, self.items_per_sample)

            with msg('Calculating posterior probabilities', enabled=self.output):
                self.posteriors = self.get_group_probs(self.group_ratings.dist(), self.user_ratings, self.samples, self.priors)
        else:
            self.posteriors = np.full((self.sample_n, self.user_n, self.priors.shape[0]), self.priors)

        self.has_not_been_run = False

    def get_accuracy(self, user_accuracy=False):
        if self.has_not_been_run: self.run()

        mean_axis = 0 if user_accuracy else 1
        return (self.posteriors.argmax(axis=2) == self.user_groups).mean(axis=mean_axis)

    def get_rmse(self, hard_memb=False):
        if self.has_not_been_run: self.run()

        with msg('Getting rmses'):
            if hard_memb:
                pred_groups = self.posteriors.argmax(axis=2)
                memb = np.zeros(self.posteriors.shape)
                memb[np.where(pred_groups > -1) + (pred_groups.flatten(),)] = 1
            else: memb = self.posteriors

            mean_ratings = self.group_ratings.mean()

            predicted_ratings = memb.dot(mean_ratings)
            errors = np.absolute(predicted_ratings - (self.user_ratings+1))
            weights = memb.dot(self.group_ratings.lam())
            return np.sqrt(np.average(errors**2, axis=(1,2), weights=weights))


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
        # group_ratings.reg(8)
        # group_ratings.thresh(25)
        # group_ratings.highest_pop(10000)
        # group_ratings.lowest_pop(1000)
        # group_ratings.lowest_var(100)
        # group_ratings.highest_var(30)
        # group_ratings.lowest_entropy(100)
        # group_ratings.highest_maxnorm(100)
        # group_ratings.highest_pnorm(100, p=1.000001)
        group_ratings.keep_n(10)
    print()

    finder = IndicatorItemsFinder(group_ratings)
    accuracy = finder.get_accuracy()

    print(f'\n{finder.user_n} users, {finder.sample_n} samples, {finder.items_per_sample} items per sample:')
    print(f'min:\t{accuracy.min():#.3}')
    print(f'max:\t{accuracy.max():#.3}')
    print(f'mean:\t{accuracy.mean():#.3}')
    print(f'median:\t{np.median(accuracy):#.3}') 
    print()   

def rmse_user_ratings_vs_Rtilde(user_ratings, user_groups, items):
    Rtilde = Data.get_Rtilde()[:,items]
    group_n = user_groups.shape[0]
    user_n = user_ratings.shape[0]

    P = np.zeros((group_n, user_n))[user_groups, np.arange(user_n)]


def plot_thresh_vs_accuracy_varying_reg():
    plt.title("Threshold vs accuracy \w 100 lowest pop")
    plt.xlabel('threshold')
    plt.ylabel('accuracy')

    for reg in [0, 1, 3, 5, 8, 13, 21, 34]: 
        count, step = 50, 2
        mean_accs = np.zeros(count)

        for i, thresh in enumerate(range(0, count*step, step)):
            group_ratings = GroupRatings()
            # group_ratings.reg(reg)
            # group_ratings.thresh(thresh)
            group_ratings.lowest_pop(100)
            # group_ratings.highest_pop(10)
            # group_ratings.lowest_var(100)
            # group_ratings.keep_n(1000)

            mean_accs[i] = IndicatorItemsFinder(item_pool).get_accuracy()

        plt.plot(np.arange(count)*step, mean_accs, label=f'reg = {reg}')

    plt.legend()
    plt.show()

# print_stats()
n_point_plots = 26
group_ratios = np.array([0, 0, 1, 0, 0, 0, 0, 0])
priors = np.ones(8) / 8
correct_error = False
a_min = np.zeros(n_point_plots)
a_max = np.zeros(n_point_plots)
a_median = np.zeros(n_point_plots)

plot_acc = False

# xs = np.arange(0, n_point_plots) * 10
xs = np.arange(0, n_point_plots) * 1
for i, x in enumerate(xs):
    g = GroupRatings(correct_error=correct_error)
    g.thresh(60)
    # g.reg(1)
    # g.highest_pop(100)
    # g.lowest_pop(100)
    # g.lowest_var(100)
    # g.lowest_entropy(100, priors)
    g.lowest_entropy(100)
    # g.highest_maxnorm(100, priors)
    # g.highest_pnorm(100)
    # g.highest_maxnorm(100)
    # g.highest_cocondition(100)
    g.keep_n(500)
    # finder = IndicatorItemsFinder(g, items_per_sample=x, user_n=100)
    # finder = IndicatorItemsFinder(g, items_per_sample=10, user_n=2000, sample_n=2000)
    # finder = IndicatorItemsFinder(g, items_per_sample=5, user_n=x, sample_n=1000)
    # finder = IndicatorItemsFinder(g, items_per_sample=5, user_n=500, sample_n=x)
    finder = IndicatorItemsFinder(g, items_per_sample=x)
    # finder = IndicatorItemsFinder(g, items_per_sample=x, group_ratios=group_ratios)
    # finder = IndicatorItemsFinder(g, items_per_sample=1, sample_n=17700, group_ratios=group_ratios, priors=priors)
    if plot_acc: accuracy = finder.get_accuracy()
    else: accuracy = finder.get_rmse()
    a_max[i] = accuracy.max()
    a_median[i] = np.median(accuracy)
    a_min[i] = accuracy.min()

plt.style.use('report')
# plt.figure(figsize=(5,3))
plt.grid(True)
# plt.xlim(0)
plt.xlabel('Number of query items')
# plt.xlabel('Number of test users')
label = 'lowest entropy'
# label = 'highest pop'
if plot_acc:
    plt.plot(xs, a_max, label=label)
    plt.plot(xs, np.load('data/baseline/highest-pop-accuracy.npy'), label='highest pop', ls='--')
    plt.ylim(0, 1.02)
    plt.ylabel('Group prediction accuracy')
    f = 'doc/report/figures/accuracy/'
else:
    plt.plot(xs, a_min, label=label)
    plt.plot(xs, np.load('data/baseline/highest-pop-rmse.npy'), label='highest pop', ls='--')
    plt.ylim(0.82, 1.12)
    plt.axhline(g.rmse(),  c='C2', ls='-.', lw=1, label=f'item pool RMSE')
    plt.ylabel('Rating prediction RMSE')
    f = 'doc/report/figures/rmse/'
plt.legend()

plot_name = f"{label.replace(' ', '-')}.png"
fname = f + plot_name
print("saving",fname)
plt.savefig(fname, bbox_inches='tight')
plt.clf()
# plt.show()
