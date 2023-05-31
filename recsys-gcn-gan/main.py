from tokenize import Name
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torch import optim
import pickle
from tqdm import tqdm
from lightfm import LightFM
import implicit
from scipy.sparse import csr_matrix
from amazmemllib.evaluation import *

from data import dataloader
from utils import *
from models.lightgcn import LightGCN
from models.lightgcn_cfg import *
from lightfm.evaluation import precision_at_k, recall_at_k

#models = {'lgcn': LightGCN, 'lfm': LightFM, 'als': implicit.als.AlternatingLeastSquares}
dataset_paths = {'ml-1m': '../data/raw/ml-1m',
                'gowalla': '../data/raw/gowalla',
                'yelp2018': '../data/raw/yelp2018',
                'amazon-books': '../data/raw/amazon-books-raw'}

@click.group()
@click.option("--seed", type=int, default=7777)
def main(seed: int) -> None:
    logging.basicConfig(level="INFO")
    logging.info("Welcome to RecSys-GCN-GAN lib!")

@main.command()
@click.option("-m", "--model", type=str, required=True)
@click.option("-d", "--dataset", type=str, required=True)
def train(model, dataset):
    """ Runs train script on selected model and dataset.
    """
    logger = logging.getLogger(__name__)
    logger.info('Let`s train!')

    dataset = dataloader.Loader(dataset_paths[dataset])
    num_users = dataset.n_users
    num_items = dataset.m_items
    print('Number of users: ', num_users)
    print('Number of items: ', num_items)
    seed = 2023
    model_name = model
    if model == 'lgcn':
        set_seed(seed)
        print("SEED: ", seed)
        model = LightGCN(dataset)
        epochs = EPOCHS
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        model.fit(optimizer=optimizer, lr_scheduler=scheduler, epochs=epochs)
        model.save_ckpt(model_name, EPOCHS)

    elif model == 'als':
        model = implicit.als.AlternatingLeastSquares(factors=50, iterations=25, regularization=0.05)
        user_item_data = dataset.UserItemData
        model.fit(user_item_data)
        model.save(f'../models/{model_name}/{model_name}_{dataset.name}.npz')
        
    elif model == 'lfm':
        model = LightFM(no_components=100, learning_schedule='adagrad', loss='warp')
        user_item_data = dataset.UserItemData
        model.fit(user_item_data, epochs=30, num_threads=4, verbose=True)
        with open(f'../models/{model_name}/{model_name}_{dataset.name}.pickle', "wb") as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        raise NameError(f'Invalid model name: {model}')

@main.command()
@click.option("-m", "--model", type=str, required=True)
@click.option("-d", "--dataset", type=str, required=True)
def evaluate(model, dataset):
    """ Runs evaluation script on selected model and dataset.
    """
    logger = logging.getLogger(__name__)
    logger.info('Evaluation')

    model_name = model
    dataset = dataloader.Loader(dataset_paths[dataset])
    num_users = dataset.n_users
    num_items = dataset.m_items

    if model == 'lgcn':
        model = LightGCN(dataset)
        #optimizer = optim.Adam(model.parameters(), lr=LR)
        model = model.to(DEVICE)
        path = f'../models/{model_name}/{model_name}_{dataset.name}.pt'
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _, recall_old, precision_old, ndcg_old = model.evaluation(TOPK, LAMBDA)
        print(f"[LightGCN Metrics] \n NDCG@20: {ndcg_old}\n Recall@20: {recall_old}\n Precision@20: {precision_old}")

    elif model == 'als':
        model = implicit.als.AlternatingLeastSquares()
        model = model.load(f'../models/{model_name}/{model_name}_{dataset.name}.npz')
        user_item_data = dataset.UserItemData
        users = []
        items = []
        test_dict = dataset.testDict
        user_ids = test_dict.keys()
        for user in user_ids:
            users.extend([user] * len(test_dict[user]))
            items.extend(test_dict[user])
        test = pd.DataFrame({'user_id': users, 'item_id': items})

        recommendations = model.recommend(list(user_ids), user_item_data[list(user_ids)], 20)
        recommendations_df = pd.DataFrame({'user_id': np.array([[i]*20 for i in user_ids]).flatten(),
                                        'item_id': recommendations[0].flatten(), 
                                        'score': recommendations[1].flatten()})
        topk = [1, 5, 10, 20]
        print(Precision(topk)(recommendations=recommendations_df, ground_truth=test))
        print(Recall(topk)(recommendations=recommendations_df, ground_truth=test))
        print(NDCG(topk)(recommendations=recommendations_df, ground_truth=test))

    elif model == 'lfm':
        with open(f'../models/{model_name}/{model_name}_{dataset.name}.pickle', "rb") as file:
            model = pickle.load(file)
        user_item_data = dataset.UserItemData
        users = []
        items = []
        topk=20
        test_dict = dataset.testDict
        user_ids = test_dict.keys()
        for user in user_ids:
            users.extend([int(user)] * len(test_dict[user]))
            items.extend(test_dict[user])
        test = pd.DataFrame({'user_id': users, 'item_id': items})
        test_sparse = csr_matrix((np.ones(len(users)), (users, items)),
                                      shape=(num_users, num_items))
                            
        precision = precision_at_k(model=model, test_interactions=test_sparse,
                    train_interactions=user_item_data, k=topk).mean()
        recall = recall_at_k(model=model, test_interactions=test_sparse,
                    train_interactions=user_item_data, k=topk).mean()
        #print(f"[LightFM Metrics] \n Recall@{topk}: {recall}\n Precision@{topk}: {precision}")

        def predict_user(user_id, topk):
            scores = model.predict(int(user_id), list(range(num_items)))
            train_items = np.nonzero(user_item_data[user_id, :])[1]
            #print('train_items ', train_items)
            scores[train_items] = -10000
            recommended_item_ids = np.argsort(-scores)[:topk]
            recommended_item_scores = scores[recommended_item_ids]
            return recommended_item_ids.tolist(), recommended_item_scores.tolist()

        pred_users, pred_items, pred_scores = [], [], []
        for user_id in tqdm(user_ids):
            recommended_item_ids, recommended_item_scores = predict_user(
                user_id, topk)
            pred_users.extend(np.repeat(user_id, len(recommended_item_ids)).tolist())
            pred_items.extend(recommended_item_ids)
            pred_scores.extend(recommended_item_scores)
        
        recommendations = pd.DataFrame({'user_id': pred_users, 'item_id': pred_items, 'score': pred_scores})
        topk = [1, 5, 10, 20]
        print('[LightFM Metrics]')
        print(Precision(topk)(recommendations=recommendations, ground_truth=test))
        print(Recall(topk)(recommendations=recommendations, ground_truth=test))
        print(NDCG(topk)(recommendations=recommendations, ground_truth=test))

    else:
        raise NameError(f'Invalid model name: {model}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
