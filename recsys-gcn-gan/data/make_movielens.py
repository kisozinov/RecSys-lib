# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from scipy import sparse

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True)) # '../../data/raw/ml-10M100K/ratings.dat'
@click.argument('output_data_filepath', type=click.Path())
@click.argument('output_target_filepath', type=click.Path())
def main(input_filepath, output_data_filepath, output_target_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        default run: python recsys-gcn-gan/data/make_movielens.py /
         data/raw/ml-10M100K/ratings.dat /
         data/interim/train.npz data/interim/test.npz  
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = pd.read_csv(input_filepath, 
                sep='::',  usecols=[0,1,2,3], 
                names=['user', 'movie', 'rating', 'timestamp'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by='timestamp', inplace=True)
    df.drop(['timestamp'], axis=1, inplace=True)
    df = df.pivot(index='user', columns='movie', values='rating').fillna(0)
    data = sparse.coo_matrix(df)
    del df

    data_arr = data.toarray()
    train, test = train_test_split(data_arr, test_size=0.5, shuffle=False)
    train_sparse = sparse.coo_matrix(train)
    test_sparse = sparse.coo_matrix(test)

    sparse.save_npz(output_data_filepath, train_sparse) # ../../data/interim/train.npz
    sparse.save_npz(output_target_filepath, test_sparse)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
