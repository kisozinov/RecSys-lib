# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from scipy import sparse
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
import pickle

@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('output_model_filepath', type=click.Path())

def main(input_data_filepath, output_model_filepath):
    """
    Runs train scripts for lightfm model.
    default run: python recsys-gcn-gan/models/lightfm_train.py /
    data/interim/train.npz models/lightfm_movielens.pickle
    """
    logger = logging.getLogger(__name__)
    logger.info('train the model')

    train_data = sparse.load_npz(input_data_filepath)

    model = LightFM(loss='warp')
    model.fit(train_data, epochs=10, num_threads=4)
    with open(output_model_filepath, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()