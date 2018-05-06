import numpy as np
import os
from group_ratings import GroupRatings
from myutils import msg

test_item_dist_means_file = 'data/testing/{}test_item_dist_means_{}_users.npy'
test_sampled_ratings_file = 'data/testing/{}sampled_ratings_{}_users.npy'
test_item_ids_file = 'data/testing/{}item_ids_{}_users.npy'

def synthesise_user_data(dists, n, group_ratios, rating_value_count, permute=True):
    """
    Synthesise n users, using the given distribution of ratings.
    NOTE: Ratings are from 0 to rating_value_count - 1
    returns 2D ratings array, and groups list corresponding to which group each user belongs to.
    """
    # distribute number of users to be generated over groups
    group_ratios = group_ratios / group_ratios.sum()
    group_sizes = (group_ratios * n).astype(int)
    remainder = n - group_sizes.sum()
    if remainder > 0:
        group_sizes[np.argpartition(group_ratios, -remainder)[-remainder:]] +=1

    ratings = []
    for group_size, group in zip(group_sizes, dists):
        if group_size != 0:
            rating_chooser = lambda p: np.random.choice(rating_value_count, size=group_size, p=p)
            ratings.append(np.stack(np.apply_along_axis(rating_chooser, 1, group).T))
    ratings = np.vstack(ratings)

    groups = np.concatenate([np.full(group_size, group) for group, group_size in enumerate(group_sizes)])

    if permute: 
        perm = np.random.permutation(ratings.shape[0])
        return ratings[perm], groups[perm]
    else: return ratings, groups


def gen_test_data(user_groups=None, item_n=100, subdir=''):
    """ Generate presistent data for testing """
    items = GroupRatings()
    item_count = items.item_count()

    user_n = user_groups.shape[0]
    test_item_means = np.zeros((user_n, items.n_groups, item_n))
    test_item_ratings = np.zeros((user_n, item_n))
    test_item_ids = np.zeros((user_n, item_n), dtype=int)

    lam_dist = items.lam() / items.lam().sum(axis=1, keepdims=True)
    dists = items.dist()
    with msg('Generating test data'):
        for n, group in enumerate(user_groups):
            # choose test items
            item_ids = np.random.choice(item_count, size=item_n, p=lam_dist[group])
            test_item_ids[n] = items.items[item_ids]

            # get group proxy ratings
            items.items = item_ids
            test_item_means[n] = items.mean()
            items.reset()

            # get sampled ratings
            for i, item_id in enumerate(item_ids):
                rating_dist = dists[group, item_id]
                test_item_ratings[n, i] = np.random.choice(items.n_rating_vals, p=rating_dist) + 1

    means_file = test_item_dist_means_file.format(subdir, user_n)
    sampled_file = test_sampled_ratings_file.format(subdir, user_n)
    item_ids_file = test_item_ids_file.format(subdir, user_n)

    with msg(f'Saving test item dist means to {means_file}'): np.save(means_file, test_item_means)
    with msg(f'Saving test item sampled ratings to {sampled_file}'): np.save(sampled_file, test_item_ratings)
    with msg(f'Saving test item ids to {item_ids_file}'): np.save(item_ids_file, test_item_ids)
    
    return test_item_means, test_item_ratings, test_item_ids


class Users:

    def __init__(self, training, user_n=1000, group=None):
        """
        groups            : the groups to which the users belong
        training_ratings  : user ratings for training
        test_item_ids     : the ids of the test items
        test_ratings      : user ratings for testing (stored to file for consistency between runs)
        test_means        : the ratings predicted for that users group
        """
        if group is not None: 
            subdir = f'group{group}_'
            group_ratios = np.zeros(training.n_groups)
            group_ratios[group] = 1
        else: 
            subdir = ''
            group_ratios = training.group_sizes()

        self.training_ratings, self.groups = synthesise_user_data(
            training.dist(), user_n, group_ratios, training.n_rating_vals, permute=False)

        means_file = test_item_dist_means_file.format(subdir, user_n)
        sampled_file = test_sampled_ratings_file.format(subdir, user_n)
        item_ids_file = test_item_ids_file.format(subdir, user_n)
        if os.path.isfile(means_file) and os.path.isfile(sampled_file) and os.path.isfile(item_ids_file):
            self.test_means = np.load(means_file)
            self.test_ratings = np.load(sampled_file)
            self.test_item_ids = np.load(item_ids_file)
        else:
            self.test_means, self.test_ratings, self.test_item_ids = gen_test_data(self.groups, subdir=subdir)

    def test_data_rmse(self):
        group_means = self.test_means[np.arange(self.test_means.shape[0]), self.groups]
        errors = np.absolute(group_means - self.test_ratings)
        return np.sqrt(np.mean(errors**2))

if __name__ == '__main__':
    g = GroupRatings()
    group=2
    d = g.lam()[group]
    d[d==0]=1
    # g.keep_n(100)
    users = Users(training=g, user_n=1, group=group)

    sm = g.get_ratings()[group].sum(axis=(0))
    tot = sm * np.arange(1,6)
    print("TOT:", tot.sum()/sm.sum())

    print(np.average(g.mean()[group], weights=g.lam()[group]))

    print(users.test_ratings.mean())
    print(users.test_means[:,group].mean())
    print(users.test_data_rmse())
