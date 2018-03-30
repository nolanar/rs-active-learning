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
Plot variance of each group by item number. The size of each bubble corresponds to the square root of the number of ratings for that distribution.
```
optional arguments:
  -h, --help         show this help message and exit
  --thresh THRESH    only plot distributions with at least thresh number of
                     ratings (defualt 50)
  --inverse          plot inverse variance instead of variance
  --savefig          save the figure to file rather than displaying the figure
  --outfile OUTFILE  file to save the figure to (default
                     "figures/variances.png")
  --rmin RMIN        Lowest index of item to display
  --rmax RMAX        Highest index of item to display (inclusive)
 ```

### `datareader.py`
Contains a utility class for getting rating and nym data.