# Accelerating Learning for Recommender Systems

Accelerating Learning for Recommender Systems via Group Learning

## Requirements
* Python 3.6+

## Setup 

1. Download the Netflix Prize data zip file from <https://www.kaggle.com/netflix-inc/netflix-prize-dat>, then put the zip file into the `data/` directory.
2. Run `./setup.sh` from the project root directory.

## Scripts

### `next_best_method.py`
Simulate the "next-best" active learning method on synthesised test users.

### `group_ratings.py`
GroupRatings class with useful methods for group ratings on items. Provides item filtering, rating distributions, rating statistics, item sampling.

### `data.py`
Utilities for reading rating and group data. Used by `group_ratings.py`.