import numpy as np
import scipy.sparse as sp
from zipfile import ZipFile
from io import TextIOWrapper

from myutils import msg

data_dir = 'data/'
ratings_file = data_dir + 'netflix_ratings.npz'
users_file = data_dir + 'netflix_orig_user_ids.npy'
movies_file = data_dir + 'netflix_orig_movie_ids.npy'

netflix_data = data_dir + 'netflix-prize-data.zip'

def parse_ratings(zipfile=netflix_data):
	filecount = 4
	basefilename = 'combined_data_{}.txt'
	ratingfiles = [basefilename.format(i) for i in range(1, filecount + 1)]

	row, col, data = [], [], []
	item_id = 1
	with msg(f'Reading from "{netflix_data}"'), ZipFile(zipfile, 'r') as myzip:
		for filename in ratingfiles:
			with msg(f'Parsing "{filename}"'), myzip.open(filename, 'r') as file:
				for line in TextIOWrapper(file):
					tokens = line.split(',')
					if len(tokens) == 3:
						row.append(int(tokens[0]))
						col.append(item_id)
						data.append(int(tokens[1]))
					else:
						item_id = int(line[:-2]) # -2 to remove ':' and newline at end of line

	with msg('Creating sparse matrix from ratings'):
		return sp.coo_matrix((data, (row, col)), dtype=np.float32)

def prepare_ratings(ratings):
	with msg('Preparing ratings'):
		with msg('Converting to csc matrix'): ratings = ratings.tocsc(copy=False)
		with msg('Removing empty cols'):
			non_zero_cols = ratings.getnnz(0) > 0
			ratings = ratings[:,non_zero_cols]

		with msg('Converting to csr matrix'): ratings = ratings.tocsr(copy=False)
		with msg('Removing empty rows'): 
			non_zero_rows = ratings.getnnz(1) > 0
			ratings = ratings[non_zero_rows]

		return ratings, np.where(non_zero_rows)[0], np.where(non_zero_cols)[0]

def save_data(ratings, user_ids, movie_ids):
	with msg(f'Saving ratings to "{ratings_file}"'):
		sp.save_npz(ratings_file, ratings)
	with msg(f'Saving original user ids to "{users_file}"'):
		np.save(users_file, user_ids)
	with msg(f'Saving original movie ids to "{movies_file}"'):
		np.save(movies_file, movie_ids)

if __name__ == '__main__':
	ratings = parse_ratings()	

	ratings, user_ids, movie_ids = prepare_ratings(ratings)
	save_data(ratings, user_ids, movie_ids)