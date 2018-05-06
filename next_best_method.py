import numpy as np
import matplotlib.pyplot as plt
from users import Users
from myutils import *
from group_ratings import GroupRatings

np.set_printoptions(3, suppress=True)

# config
user_n_pre_group = 100 # number of test users per group 
default_thresh = 100
default_quota = 20

def get_rmse(prediction, test_ratings):
    return np.sqrt(np.mean((prediction - test_ratings)**2, axis=2))

def entropy(x, axis, keepdims=False):
    """ NOTE: returns negative entropy as max is prioritised for utility """
    with suppress_numpy_err():
        x *= np.log(x)
        x[np.isnan(x)] = 0    
    return x.sum(axis=axis, keepdims=keepdims)

def maxnorm(x, axis, keepdims=False):
    return x.max(axis=axis, keepdims=keepdims)

def pnorm(x, axis, keepdims=False, p=2):
    return ((x**p).sum(axis=axis, keepdims=keepdims))**(1/p)

def utility(priors, liklihoods, util_f):
    posteriors = liklihoods * priors[:,:,None,None]  # n x p x m x r
    prob_R = posteriors.sum(axis=1, keepdims=True)   # n x 1 x m x r
    posteriors /= prob_R                             # n x p x m x r
    util = util_f(posteriors, axis=1, keepdims=True) # n x 1 x m x r
    return np.squeeze(util * prob_R).sum(axis=2)     # n x m

def run(group, quota=default_quota, user_n=user_n_pre_group, info=True, util_f=entropy, util_label='entropy', thresh=default_thresh):
    '''
    Run simulation of "next-best" active learning method.

    group       = group to which the test users belong
    quota       = max number of items to be queried
    user_n      = number of test users
    info        = print running information
    util_f      = function to use as measure of item utility (max values prioritised)
    util_label  = utility measure function label, for saving results to file
    thresh      = item rating threshold (see GroupRatings.thresh method)
    '''

    # get test users
    items = GroupRatings()
    items.thresh(thresh)

    with msg(f'Generating training data for {user_n} test users on {items.item_count()} items'):
        test_users = Users(training=items, user_n=user_n, group=group)
        test_user_groups = test_users.groups            # n
        training_ratings = test_users.training_ratings  # n x m_train
        test_means = test_users.test_means              # n x p x m_test
        test_ratings = test_users.test_ratings          # n x m_test

    size = quota+1
    queried_items = np.zeros((user_n, training_ratings.shape[1]), dtype=bool) # mask items that have been quered
    group_probs = np.zeros((size, user_n, items.n_groups)) # size x n x p
    group_probs[0] = items.group_size_dist() # initial group probs
    liklihoods = items.dist() # p x m x r

    for q in range(1, size):
        with msg(f'group {group}, query {q}'):
            priors = group_probs[q-1]
            # get items to query
            item_utils = np.ma.array(utility(priors, liklihoods, util_f), mask=queried_items)
            query_items = np.argmax(item_utils, axis=1) # n
            queried_items[np.arange(queried_items.shape[0]), query_items] = True

            # update priors using response to query
            responses = training_ratings[np.arange(training_ratings.shape[0]), query_items] # n
            posteriors = liklihoods[:,query_items,responses].T * priors # n x p
            posteriors /= posteriors.sum(axis=1, keepdims=True)
            group_probs[q] = posteriors

    # get group prediction accuracy and rating prediction RMSE
    pred_groups = np.argmax(group_probs, axis=2) # size x n
    pred_group_accuracy = (test_user_groups == pred_groups).mean(axis=1) # size
    prediction_soft = (group_probs[:,:,:,None] * test_means).sum(axis=2) # size x n x m_test
    prediction_hard = test_means[np.arange(pred_groups.shape[1]), pred_groups] # size x n x m_test
    soft_pred_rmse = get_rmse(prediction_soft, test_ratings) # size x n
    hard_pred_rmse = get_rmse(prediction_hard, test_ratings) # size x n

    with msg('Saving predictions', enabled=info):
        np.save(f'data/next-best/pred_group_{group}{util_label}.npy', pred_groups)
        np.save(f'data/next-best/pred_group_accuracy_{group}{util_label}.npy', pred_group_accuracy)
        np.save(f'data/next-best/soft_pred_rmse_{group}{util_label}.npy', soft_pred_rmse)
        np.save(f'data/next-best/hard_pred_rmse_{group}{util_label}.npy', hard_pred_rmse)

def plot_rmse(group, util_label='entropy'):
    def plot_one_memb(pred_rmses, tag, ls='-'):
        min_rmses = np.min(pred_rmses, axis=1)
        median_rmses = np.median(pred_rmses, axis=1)
        max_rmses = np.max(pred_rmses, axis=1)
        xs = np.arange(pred_rmses.shape[0])
        plt.plot(xs, min_rmses, c='C0', label=f'{tag} min', ls=ls)
        plt.plot(xs, median_rmses, c='C1', label=f'{tag} median', ls=ls)
        plt.plot(xs, max_rmses, c='C2', label=f'{tag} max', ls=ls)

    with msg('loading data'):
        soft_pred_rmses = np.load(f'data/next-best/soft_pred_rmse_{group}{util_label}.npy')
        hard_pred_rmses = np.load(f'data/next-best/hard_pred_rmse_{group}{util_label}.npy')

    plot_one_memb(soft_pred_rmses, 'soft')
    plot_one_memb(hard_pred_rmses, 'hard', ls='-.')

    plt.xlabel('Number of queries')
    plt.xlabel('Prediction RMSE')

    plt.legend()
    plt.show()

def plot_group_accuracy(group, util_label='entropy'):
    pred_group_accuracy = np.load(f'data/next-best/pred_group_accuracy_{group}{util_label}.npy')

    xs = np.arange(pred_group_accuracy.shape[0])
    plt.plot(xs, pred_group_accuracy)
    plt.ylim([0,1.02])

    plt.xlabel('Number of queries')
    plt.xlabel('Group prediction accuracy')

    plt.legend()
    plt.show()

def plot_accuracy_all_groups(util_label='entropy', savefig=False):
    g = GroupRatings()
    group_size_dist = g.group_size_dist()
    plt.figure(figsize=(10,6))
    for group in range(g.n_groups):
        pred_group_accuracy = np.load(f'data/next-best/pred_group_accuracy_{group}{util_label}.npy')
        xs = np.arange(pred_group_accuracy.shape[0])
        plt.plot(xs, pred_group_accuracy, label=f'group {group}')
        if group == 0: total_accuracy = group_size_dist[group]*pred_group_accuracy
        else: total_accuracy += group_size_dist[group]*pred_group_accuracy
    
    xs = np.arange(total_accuracy.shape[0])
    plt.plot(xs, total_accuracy, label=f'combined', c='black', ls='--')

    # plt.xlim([0-0.1, total_accuracy.shape[0]-0.9])
    plt.ylim([0,1.02])
    plt.xlabel('Number of queries')
    plt.ylabel('Group prediction accuracy')
    plt.xticks(xs)
    plt.yticks(np.arange(11)/10)
    plt.legend()

    if util_label == '': tag = 'entropy'
    else: tag = util_label
    if savefig: 
        f = f'figures/next-best-accuracy-{tag}'
        with msg(f"saving {f}"):
            plt.savefig(f)
    else: plt.show()

def plot_rmse_all_groups(util_label='entropy', savefig=False, memb='soft'):
    g = GroupRatings()
    group_size_dist = g.group_size_dist()
    plt.figure(figsize=(10,6))
    for group in range(g.n_groups):
        pred_rmse = np.load(f'data/next-best/{memb}_pred_rmse_{group}{util_label}.npy')
        pred_mse_total = (pred_rmse**2).mean(axis=1)
        pred_rmse_total = pred_mse_total**(1/2)
        xs = np.arange(pred_rmse_total.shape[0])
        plt.plot(xs, pred_rmse_total, label=f'group {group}')
        if group == 0: total_mse = group_size_dist[group]*pred_mse_total
        else: total_mse += group_size_dist[group]*pred_mse_total
    total_rmse = (total_mse)**(1/2)
    
    xs = np.arange(total_rmse.shape[0])
    plt.plot(xs, total_rmse, label=f'combined', c='black', ls='--')

    total_test_rmse, group_rmse = test_data_rmse()
    plt.axhline(total_test_rmse,  c='b', ls='-.', label=f'target RMSE')

    plt.xlabel('Number of queries')
    plt.ylabel('Rating prediction RMSE')
    plt.xticks(xs)
    plt.yticks(np.linspace(0.75, 1.35, 13))
    plt.ylim(0.75, 1.35)
    plt.legend()

    if util_label == '': tag = 'entropy'
    else: tag = util_label
    if savefig: 
        f = f'figures/next-best-{memb}-rmse-{tag}'
        with msg(f"saving {f}"): 
            plt.savefig(f)
    else: plt.show()


def test_data_rmse(user_n_per_group=user_n_pre_group):
    """ Get RMSE of test data over all groups with "user_n_per_group" test users per group """
    g = GroupRatings()
    g.keep_n(1)
    total_mse = 0
    group_size_dist = g.group_size_dist()
    group_rmse = np.zeros(g.n_groups)
    for group in range(g.n_groups): 
        test_users = Users(training=g, user_n=user_n_per_group, group=group)
        total_mse += test_users.test_data_rmse()**2 * group_size_dist[group]
        group_rmse[group] = test_users.test_data_rmse()
    total_rmse = total_mse**(1/2)
    return total_rmse, group_rmse

if __name__ == '__main__':
    with msg("Running and plotting with util_f=entropy, util_label='entropy'"):
        for group in range(GroupRatings().n_groups): run(group)
        plot_group_accuracy(group=0)
        plot_rmse(group=0)
        plot_accuracy_all_groups()
        plot_rmse_all_groups()