# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from scipy import sparse
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k, recall_at_k
import pickle

@click.command()
@click.argument('input_model_filepath', type=click.Path(exists=True))
@click.argument('input_data_filepath', type=click.Path())
#@click.argument('output_sample_filepath', type=click.Path())

def main(input_model_filepath, input_data_filepath, output_sample_filepath=None):
    """
    Returns the prediction of model.
    defaul run: python recsys-gcn-gan/models/lightfm_predict.py /
    models/lightfm_movielens.pickle /
    data/interim/test.npz /

    """
    logger = logging.getLogger(__name__)
    logger.info('train the model')

    with open(input_model_filepath, "rb") as input_file:
        model = pickle.load(input_file)

    test_data = sparse.load_npz(input_data_filepath)
    test_precision = precision_at_k(model, test_data, k=5).mean()
    test_recall = recall_at_k(model, test_data, k=5).mean()
    print(f"Precision at 5 is: {test_precision}")
    print(f"Recall at 5 is: {test_recall}")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()