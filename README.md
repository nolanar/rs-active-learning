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

### `plot_variances.py`
Plot variance of each group by item number. The size of each bubble corresponds to the square root of the number of ratings for that distribution. Only bubbles with at least the threshold number of ratings are plotted.
```
optional arguments:
  -h, --help        show this help message and exit
  -b                index of the item to begin plotting from
  -n                number of items to plot
  -t                only plot distributions with at least threshold number of
                    ratings (defualt 50)
  -i                plot inverse variance instead of variance
  --savefig         save the figure to file rather than displaying the figure
  -outfile          file to save the figure to (default
                    "figures/variances.png")
 ```

### `datareader.py`
Contains a utility class for getting rating and nym data.