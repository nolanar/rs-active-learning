import numpy as np
import matplotlib.pyplot as plt

from myutils import msg
from users import Users
from group_ratings import GroupRatings

# config
default_n_points = 26

def accuracy_save_file(label, n_points):
    return f'data/sampling/{label.replace(" ", "-")}-accuracy-{n_points}.npy'

def rmse_save_file(label, n_points, hard_memb=False):
    hard = '-hard' if hard_memb else ''
    return f'data/sampling/{label.replace(" ", "-")}-rmse-{n_points}{hard}.npy'

def get_posteriors(group_ratings_dist, user_ratings, samples, priors):
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

def get_accuracy(posteriors, users):
    accuracies = (posteriors.argmax(axis=2) == users.groups).mean(axis=1)
    return accuracies

def get_rmse(posteriors, users, hard_memb=False):
    if hard_memb:
        pred_groups = posteriors.argmax(axis=2)
        memb = np.zeros(posteriors.shape)
        memb[np.where(pred_groups > -1) + (pred_groups.flatten(),)] = 1
    else: memb = posteriors

    predicted_ratings = (np.expand_dims(memb, -1) * users.test_means).sum(axis=2)
    errors = np.absolute(predicted_ratings - users.test_ratings)
    rmses = np.sqrt(np.mean(errors**2, axis=(1,2)))
    return rmses

def save_fig(f, label, tag=None):
    if tag is None: tag = ''
    else: tag = f'-{tag}'
    plot_name = f"{label.replace(' ', '-')}{tag}.png"
    fname = f + plot_name
    with msg("saving",fname):
        plt.savefig(fname, bbox_inches='tight')
        plt.clf()

def plot_rmse(min_rmses, true_rmse, label, baseline=None, savefig=False, hard_memb=False):
    plt.grid(True)
    plt.xlabel('Number of items queried')
    plt.ylabel('Rating prediction RMSE')
    
    n_points = min_rmses.shape[0]
    xs = np.arange(n_points)
    plt.plot(xs, min_rmses, label=label)
    if baseline is not None: plt.plot(xs, np.load(rmse_save_file('passive', n_points, hard_memb=hard_memb)), label='passive', ls='--')
    plt.axhline(true_rmse,  c='C2', ls='-.', lw=1, label=f'target RMSE')
    plt.legend()
    
    if savefig: save_fig('figures/sampling-rmse-', label, 'hard' if hard_memb else None)
    else: plt.show()
    
def plot_accuracy(max_accuracies, label, baseline=True, savefig=False):
    plt.grid(True)
    plt.xlabel('Number of items queried')
    plt.ylabel('Group prediction accuracy')
    plt.ylim(0, 1.02)

    n_points = max_accuracies.shape[0]
    xs = np.arange(n_points)
    plt.plot(xs, max_accuracies, label=label)
    if baseline: plt.plot(xs, np.load(accuracy_save_file('passive', n_points)), label='passive', ls='--')
    plt.legend()
    
    if savefig: save_fig('figures/sampling-accuracy-', label)
    else: plt.show()

def plot_rmse_spread(min_rmses, median_rmses, max_rmses, true_rmse, label, savefig=False, hard_memb=False):
    plt.grid(True)
    plt.xlabel('Number of items queried')
    plt.ylabel('Rating prediction RMSE')
    plt.ylim(0.88, 1.10)
    
    n_points = min_rmses.shape[0]
    xs = np.arange(n_points)
    plt.plot(xs, median_rmses, label='median')
    plt.plot(xs, max_rmses, label='max', ls='--')
    plt.plot(xs, min_rmses, label='min', ls='--')
    plt.axhline(true_rmse,  c='C3', ls='-.', lw=1, label=f'target RMSE')
    plt.legend()
    
    if savefig: save_fig('figures/sampling-rmse-', label, 'hard' if hard_memb else None)
    else: plt.show()
    
def plot_accuracy_spread(min_accuracies, median_accuracies, max_accuracies, label, savefig=False):
    plt.grid(True)
    plt.xlabel('Number of items queried')
    plt.ylabel('Group prediction accuracy')
    plt.ylim(0, 1.02)

    n_points = max_accuracies.shape[0]
    xs = np.arange(n_points)
    plt.plot(xs, median_accuracies, label='median')
    plt.plot(xs, max_accuracies, label='max', ls='--')
    plt.plot(xs, min_accuracies, label='min', ls='--')
    plt.legend()
    
    if savefig: save_fig('figures/sampling-accuracy-', label)
    else: plt.show()

def run(label=None, user_n=500, sample_n=500, n_points=default_n_points, correct_error=False, thresh=60, best_n=100, with_rmse=True, 
        with_accuracy=True, save_points=True, baseline=True, weight=False, plot_spread=False, plot=True, hard_memb=False):

    # pick item pool
    with msg('Configuring item pool'):
        g = GroupRatings()
        g.thresh(thresh)

        if label is None: label = 'sampling by pop'
        if label == 'highest pop':      g.highest_pop(best_n)
        if label == 'lowest pop':       g.lowest_pop(best_n)
        if label == 'highest variance': g.highest_var(best_n)
        if label == 'lowest variance':  g.lowest_var(best_n)
        if label == 'lowest entropy':   g.lowest_entropy(best_n)
        if label == 'highest 2-norm':   g.highest_pnorm(best_n)
        if label == 'highest max-norm': g.highest_maxnorm(best_n);

    # generate user data
    with msg('Generating users'):
        users = Users(training=g, user_n=user_n)

    priors = g.group_size_dist()
    liklihoods = g.dist()
    max_accuracies = np.zeros(n_points)
    median_accuracies = np.zeros(n_points)
    min_accuracies = np.zeros(n_points)
    max_rmses = np.zeros(n_points)
    median_rmses = np.zeros(n_points)
    min_rmses = np.zeros(n_points)
    if hard_memb:
        max_rmses_hard = np.zeros(n_points)
        median_rmses_hard = np.zeros(n_points)
        min_rmses_hard = np.zeros(n_points)
    for point in range(0, n_points):
        with msg(f'Computing point {point}'):
            with msg('Getting posteriors'):
                if point > 0:
                    samples = g.sample(min(sample_n, g.item_count()**point), items_per=point, weight=weight)
                    posteriors = get_posteriors(liklihoods, users.training_ratings, samples, priors)
                else:
                    posteriors = np.full((1, user_n, priors.shape[0]), priors)

            if with_accuracy: 
                with msg('Getting max accuracy'): 
                    accuracies = get_accuracy(posteriors, users)
                    max_accuracies[point] = np.max(accuracies)
                    median_accuracies[point] = np.median(accuracies)
                    min_accuracies[point] = np.min(accuracies)

            if with_rmse: 
                with msg('Getting min RMSE'): 
                    rmses = get_rmse(posteriors, users, hard_memb=False)
                    max_rmses[point] = np.max(rmses)
                    median_rmses[point] = np.median(rmses)
                    min_rmses[point] = np.min(rmses)

                if hard_memb:
                    with msg('Getting min RMSE hard memeb'): 
                        rmses_hard = get_rmse(posteriors, users, hard_memb=True)
                        max_rmses_hard[point] = np.max(rmses_hard)
                        median_rmses_hard[point] = np.median(rmses_hard)
                        min_rmses_hard[point] = np.min(rmses_hard)

    if save_points:
        if label == 'sampling by pop':
            plabel = 'passive'
            if with_accuracy: 
                with msg('saving', accuracy_save_file(plabel, n_points)):
                    np.save(accuracy_save_file(plabel, n_points), median_accuracies)
            if with_rmse: 
                with msg('saving', rmse_save_file(plabel, n_points)):
                    np.save(rmse_save_file(plabel, n_points), median_rmses)
            if hard_memb and with_rmse:
                    with msg('saving', rmse_save_file(plabel, n_points, hard_memb=True)):
                        np.save(rmse_save_file(plabel, n_points, hard_memb=True), median_rmses_hard)
        else:
            if with_accuracy: 
                with msg('saving', accuracy_save_file(label, n_points)):
                    np.save(accuracy_save_file(label, n_points), max_accuracies)
            if with_rmse: 
                with msg('saving', rmse_save_file(label, n_points)):
                    np.save(rmse_save_file(label, n_points), min_rmses)
            if hard_memb and with_rmse: 
                with msg('saving', rmse_save_file(label, n_points, hard_memb=True)):
                    np.save(rmse_save_file(label, n_points, hard_memb=True), min_rmses_hard)


    if plot_spread:
        if with_accuracy: plot_accuracy_spread(min_accuracies, median_accuracies, max_accuracies, label)
        if with_rmse: plot_rmse_spread(min_rmses, median_rmses, max_rmses, users.test_data_rmse(), label, hard_memb=False)        
        if hard_memb:
            if with_rmse: plot_rmse_spread(min_rmses_hard, median_rmses_hard, max_rmses_hard, users.test_data_rmse(), label, hard_memb=True)        
    elif plot:
        if with_accuracy: plot_accuracy(max_accuracies, label, baseline)
        if with_rmse: plot_rmse(min_rmses, users.test_data_rmse(), label, baseline, hard_memb=False)
        if hard_memb:
            if with_rmse: plot_rmse(min_rmses_hard, users.test_data_rmse(), label, baseline, hard_memb=True)

labels = ['sampling by pop', 'highest pop', 'lowest pop', 'highest variance', 'lowest variance', 'lowest entropy', 'highest 2-norm', 'highest max-norm']
lss = ['-', ':', ':', '-.', '-.', '-', '-', '-']
def plot_all(n_points=default_n_points, option='accuracy', savefig=False):
    """ option = 'accuracy' or 'rmse' """
    plt.figure(figsize=(10,7))
    plt.grid(True)
    plt.xlabel('Number of items queried')
    if option == 'accuracy': 
        plt.ylabel('Group prediction accuracy')
        plt.ylim(0, 1.02)
    if option == 'rmse': 
        plt.ylabel('Rating prediction RMSE')

    xs = np.arange(n_points)
    for label, ls in zip(labels, lss):
        if option == 'accuracy': ys = np.load(accuracy_save_file(label, n_points))
        if option == 'rmse': ys = np.load(rmse_save_file(label, n_points))
        plt.plot(xs, ys, label=label, ls=ls)
    
    if option == 'accuracy': passive_ys = np.load(accuracy_save_file('passive', n_points))
    if option == 'rmse': passive_ys = np.load(rmse_save_file('passive', n_points))
    plt.plot(xs, passive_ys, label='passive', ls='--', color='black')

    if option == 'rmse':
        g = GroupRatings(output=False)
        g.keep_n(1)
        users = Users(training=g, user_n=500)
        plt.axhline(users.test_data_rmse(),  c='b', ls='-.', label=f'target RMSE')
    
    plt.legend()
    if savefig: save_fig(f'figures/sampling-{option}-', 'all-heuristics')
    else: plt.show()

if __name__ == '__main__':
    # for label in labels:
    #     with msg(f"Running sampling method for '{label}' heuristic"):
    #         run(label=label, plot=False)

    # plot_all(option='accuracy')
    plot_all(option='rmse')