#!/usr/bin/env bash

# install reqiured python libraries
pip install -r requirements.txt

# create required empty directories
mkdir -p "figures/"
mkdir -p "data/dir_cache/"

# unzip data
netflix_data_files=(
	'data/netflix_ratings.npz'
	'data/netflix_orig_user_ids.npy'
	'data/netflix_orig_movie_ids.npy'
)
netflix_data_zip="data/netflix-prize-data.zip"
netflix_data_source="https://www.kaggle.com/netflix-inc/netflix-prize-data"

has_nf_files=true
for nf_file in "${netflix_data_files[@]}"; do
	if [ ! -f $nf_file ]; then has_nf_files=false; fi
done
if ! $has_nf_files; then
	if [ ! -f $netflix_data_zip ]; then
		echo "\"$netflix_data_zip\" not found. Download from $netflix_data_source and store into data directory, then run this script again."
		exit 1
	fi

	echo "Processing \"$netflix_data_zip\" using netflix_data.py. This may take a few minutes to complete."
	python3 netflix_data.py;
	ret=$?
	if [ $ret -eq 0 ]; then 
		echo "Done processing $netflix_data_zip"
	else
		echo "A problem occured while running netflix_data.py. Now exiting setup script."
		exit $ret
	fi
fi

echo "Setup complete!"