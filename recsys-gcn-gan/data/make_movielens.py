# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
import scipy
from scipy import sparse

from sklearn import preprocessing

import preprocess_utils

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True)) # '../../data/raw/ml-10M100K/ratings.dat'
@click.argument('output_train_filepath', type=click.Path())
@click.argument('output_test_filepath', type=click.Path())
def main(input_filepath, output_train_filepath, output_test_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        default run: python recsys-gcn-gan/data/make_movielens.py /
         data/raw/ml-10M100K/ratings.dat /
         data/interim/train.npz data/interim/test.npz  
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data_ratings = pd.read_csv(input_filepath, 
                sep='::',  usecols=[0,1,2,3], 
                names=['user', 'movie', 'rating', 'timestamp'])

    data_ratings = preprocess_utils.mark_last_n_ratings_as_validation_set(data_ratings, 5)

    # Encode users and movies id's
    le_users = preprocessing.LabelEncoder()
    le_movies = preprocessing.LabelEncoder()
    le_users = le_users.fit(data_ratings.user)
    le_movies = le_movies.fit(data_ratings.movie)

    train_data = data_ratings[data_ratings.is_valid==False]
    train_data.user = le_users.transform(train_data.user)
    train_data.movie = le_movies.transform(train_data.movie)
    train_data = train_data.sort_values('user')

    test_data = data_ratings[data_ratings.is_valid==True]
    test_data.user = le_users.transform(test_data.user)
    test_data.movie = le_movies.transform(test_data.movie)
    test_data = test_data.sort_values('user')

    # Make sparse matrices in COOrdinate format
    train_sparse = sparse.coo_matrix((train_data['rating'], (train_data['user'], train_data['movie'])),
        shape=(train_data.user.max()+1, train_data.movie.max()+1))
    
    test_sparse = sparse.coo_matrix((test_data['rating'], (test_data['user'], test_data['movie'])),
        shape=(test_data.user.max()+1, test_data.movie.max()+1))


    sparse.save_npz(output_train_filepath, train_sparse) # ../../data/interim/train.npz
    sparse.save_npz(output_test_filepath, test_sparse)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
