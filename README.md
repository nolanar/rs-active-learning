# Active Learning for Recommender Systems

## Setup 
Run `./setup.sh`
* installs required python libraries
* creates empty directories

## Scripts

### `plot_variances.py`
Plot vairance of each group by item number
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