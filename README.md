# Active Learning for Recommender Systems

Active Learning for Recommender Systems via Group Learning

## Requirements
* Python 3.6+

## Setup 

Run `./setup.sh`
* installs required python libraries
* creates empty directories
* unzips data

## Scripts

### `plot_nym_stat.py`
Plot mean or variance of each group by item number. The size of each bubble corresponds to the square root of the number of ratings for that distribution. Only bubbles with at least the threshold number of ratings are plotted.
```
optional arguments:
  -h, --help         show this help message and exit
  -o                 1 to plot mean, 2 to plot variance, (default 2)
  -b                 index of the item to begin plotting from
  -n                 number of items to plot
  -t                 only plot distributions with at least threshold number of
                     ratings (defualt 50)
  -i                 plot inverse of chosen stat instead
  --savefig          save the figure to file rather than displaying the figure
  --outfile          file to save the figure to (default
                     "figures/nym_stat.png")
 ```

### `datareader.py`
Contains a utility class for getting rating and nym data. 

Change the uncommented `blc_data` at the top of the config section to select which BLC generated data (nyms, V, Utilde etc.) is used.