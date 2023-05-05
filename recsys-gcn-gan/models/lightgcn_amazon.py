from collections import UserList
import random
from turtle import update
from typing import Dict, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import torch
from torch import nn, optim, Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.graphgym import save_ckpt, set_run_dir
from torch_geometric.graphgym.config import cfg
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from pyspark.sql import DataFrame as SparkDataFrame

from amazmemllib.utils import get_spark_session, init_logger
from amazmemllib.preprocessing import InteractionEntriesFilter, CSRConverter, LabelEncoder, LabelEncodingRule
from amazmemllib.evaluation import *

from spark import get_local_spark_session

init_logger(level="DEBUG")

MLSPACE_S3_BUCKET = 'b-ws-a7qu1-pd12-kd9'
os.environ['ADVANCED_S3_ENDPOINT'] = ''
os.environ['ADVANCED_ACCESS_KEY_ID'] = ''
os.environ['ADVANCED_SECRET_ACCESS_KEY'] = ''

os.environ['MLSPACE_S3_ENDPOINT'] = 'https://n-ws-a7qu1-pd12.s3pd12.sbercloud.ru'
os.environ['MLSPACE_ACCESS_KEY_ID'] =  'u-ws-a7qu1-pd12-l4x'
os.environ['MLSPACE_SECRET_ACCESS_KEY'] = 'w0icDRyF5tOme0sHm0gepmvQiaXbZ2Z2aHn9JTai'
os.environ['MLSPACE_S3_BUCKET'] = MLSPACE_S3_BUCKET

spark = get_local_spark_session(memory='450g')

dataset_name = 'beauty'

set_run_dir(f'../models/lightgcn/amazon_{dataset_name}')

train_events = spark.read.parquet(f'/home/jovyan/.art/cache/datasets/amazon/{dataset_name}/last_one_out/train_events.parquet')
validation_gt = spark.read.parquet(f'/home/jovyan/.art/cache/datasets/amazon/{dataset_name}/last_one_out/validation_ground_truth.parquet')
test_gt = spark.read.parquet(f'/home/jovyan/.art/cache/datasets/amazon/{dataset_name}/last_one_out/test_ground_truth.parquet')
test_events = spark.read.parquet(f'/home/jovyan/.art/cache/datasets/amazon/{dataset_name}/last_one_out/test_events.parquet')
validation_events = spark.read.parquet(f'/home/jovyan/.art/cache/datasets/amazon/{dataset_name}/last_one_out/validation_events.parquet')
#user_features = spark.read.parquet(f'/home/jovyan/.art/cache/datasets/movielens/1m/last_one_out/user_features.parquet')
#item_features = spark.read.parquet(f'/home/jovyan/.art/cache/datasets/movielens/1m/last_one_out/item_features.parquet')

train_events.count()

# load user and movie nodes
def load_node_dat(
    df: SparkDataFrame,
    col_name : str
    ) -> Dict[int, int]:
    """Loads .dat containing node information
    Args:
        df (SparkDataFrame): Pyspark dataframe with interactions
        col_name (str): column name of index column

    Returns:
        dict: mapping of .dat row to node id
    """
    mapping = {index.__getitem__(col_name): i for i, index in enumerate(df.select(col_name).distinct().collect())}
    return mapping

# load edges between users and movies
def load_edge_dat(
    df: SparkDataFrame,
    src_index_col: str,
    src_mapping: Dict[int, int],
    dst_index_col: str,
    dst_mapping: Dict[int, int],
    link_index_col: str
    ) -> torch.Tensor:
    """Loads csv containing edges between users and items

    Args:
        df (SparkDataFrame): Pyspark dataframe with interactions
        src_index_col (str): column name of users
        src_mapping (dict): mapping between row number and user id
        dst_index_col (str): column name of items
        dst_mapping (dict): mapping between row number and item id
        link_index_col (str): column name of user item interaction
        rating_threshold (int, optional): Threshold to determine positivity of edge. Defaults to 4.

    Returns:
        torch.Tensor: 2 by N matrix containing the node ids of N user-item edges
    """
    edge_index = None
    df = df.toPandas()
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    #edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long)

    edge_index = [[], []]
    for i in range(df.shape[0]):
        edge_index[0].append(src[i])
        edge_index[1].append(dst[i])
    return torch.tensor(edge_index)


def sample_mini_batch(
    batch_size: int,
    edge_index: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        batch_size (int): minibatch size
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        tuple: user indices, positive item indices, negative item indices
    """
    num_nodes = (edge_index[1].max()+1).cpu()
    edges = structured_negative_sampling(edge_index, num_nodes=num_nodes) # edited
    #print('After sampling:', edges)
    edges = torch.stack(edges, dim=0)
    #print('After stack:', edges)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices

# defines LightGCN model
class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users, num_items, embedding_dim=128, K=16, add_self_loops=False):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        edge_index_norm = gcn_norm(
            edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0
        embs = [emb_0]
        emb_k = emb_0

        # multi-scale diffusion
        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1) # E^K

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)


def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val):
    """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

    Args:
        users_emb_final (torch.Tensor): e_u_k
        users_emb_0 (torch.Tensor): e_u_0
        pos_items_emb_final (torch.Tensor): positive e_i_k
        pos_items_emb_0 (torch.Tensor): positive e_i_0
        neg_items_emb_final (torch.Tensor): negative e_i_k
        neg_items_emb_0 (torch.Tensor): negative e_i_0
        lambda_val (float): lambda value for regularization loss term

    Returns:
        torch.Tensor: scalar bpr loss value
    """
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2)) # L2 loss

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1) # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1) # predicted scores of negative samples

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss

def get_user_positive_items(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
        
    return user_pos_items


# wrapper function to get evaluation metrics
def get_metrics(model, edge_index, exclude_edge_indices, k, ground_truth_df):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    # get ratings between every user and item - shape is num users x num movies
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)
        
        # set ratings of excluded edges to large negative value
        rating[exclude_users, exclude_items] = -(1 << 10)
    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k) 
    # get all unique users in evaluated split
    test_user_pos_items = get_user_positive_items(edge_index)
    users = edge_index[0].unique()
    scores = []
    for user, items in zip(users, top_K_items):
        for item in items:
            scores.append([user.item(), item.item(), rating[user, item].item()])

    recommendations = pd.DataFrame(scores, columns=['user_id', 'item_id', 'score']).sort_values('user_id')

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = []
    for user in users:
        test_user_pos_items_list.append(test_user_pos_items[user.item()])
    ground_truth_df = ground_truth_df.toPandas()[['user_id', 'item_id']]#.sort_values('user_id')
    #ground_truth_df['user_id'] = ground_truth_df['user_id'].apply(lambda x: x-1) # CHECK
    # NEED INVERSE MAPPING
    ground_truth_df = ground_truth_df.replace({'user_id':user_mapping}).replace({'item_id':movie_mapping})
    map_ = MAP(k)(recommendations=recommendations, ground_truth=ground_truth_df)
    ndcg = NDCG(k)(recommendations=recommendations, ground_truth=ground_truth_df)
    recall = Recall(k)(recommendations=recommendations, ground_truth=ground_truth_df)
    precision = Precision(k)(recommendations=recommendations, ground_truth=ground_truth_df)

    return map_, ndcg, recall, precision

# wrapper function to evaluate model
def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, topk, lambda_val, ground_truth_df):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """
    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index)
    num_nodes = (edge_index[1].max()+1).cpu()
    edges = structured_negative_sampling(
        edge_index, num_nodes=num_nodes, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
        pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[
        neg_item_indices], items_emb_0[neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    map_dict, ndcg_dict, recall_dict, precision_dict = {},{},{},{},
    for k in topk:
        map_, ndcg, recall, precision = get_metrics(
            model, edge_index, exclude_edge_indices, k, ground_truth_df)
        map_dict.update(map_)
        ndcg_dict.update(ndcg)
        recall_dict.update(recall)
        precision_dict.update(precision)
        
    # round values
    map_dict = {key : round(map_dict[key], 5) for key in map_dict}
    ndcg_dict = {key : round(ndcg_dict[key], 5) for key in ndcg_dict}
    precision_dict = {key : round(precision_dict[key], 5) for key in precision_dict}
    recall_dict = {key : round(recall_dict[key], 5) for key in recall_dict}

    return loss, map_dict, ndcg_dict, recall_dict, precision_dict

all_events = train_events.union(test_gt).union(validation_gt)
user_mapping = load_node_dat(all_events, col_name='user_id')
movie_mapping = load_node_dat(all_events, col_name='item_id')

num_users, num_movies = len(user_mapping), len(movie_mapping)

train_edge_index = load_edge_dat(
    train_events,
    src_index_col='user_id',
    src_mapping=user_mapping,
    dst_index_col='item_id',
    dst_mapping=movie_mapping,
    link_index_col='rating',
)
val_edge_index = load_edge_dat(
    validation_gt,
    src_index_col='user_id',
    src_mapping=user_mapping,
    dst_index_col='item_id',
    dst_mapping=movie_mapping,
    link_index_col='rating',
)
test_edge_index = load_edge_dat(
    test_gt,
    src_index_col='user_id',
    src_mapping=user_mapping,
    dst_index_col='item_id',
    dst_mapping=movie_mapping,
    link_index_col='rating',
)
val_exclude_edge_index = load_edge_dat(
    validation_events,
    src_index_col='user_id',
    src_mapping=user_mapping,
    dst_index_col='item_id',
    dst_mapping=movie_mapping,
    link_index_col='rating',
)
test_exclude_edge_index = load_edge_dat(
    test_events,
    src_index_col='user_id',
    src_mapping=user_mapping,
    dst_index_col='item_id',
    dst_mapping=movie_mapping,
    link_index_col='rating',
)
# convert edge indices into Sparse Tensors: https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))
val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))
test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))



model = LightGCN(num_users, num_movies)

# define contants
ITERATIONS = 3000
BATCH_SIZE = 128
LR = 1e-3
ITERS_PER_EVAL = 200
ITERS_PER_LR_DECAY = 200
TOPK = [1, 5, 10]
LAMBDA = 1e-6

user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
        BATCH_SIZE, train_edge_index)

# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")


model = model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

#edge_index = edge_index.to(device)
train_edge_index = train_edge_index.to(device)
train_sparse_edge_index = train_sparse_edge_index.to(device)
val_edge_index = val_edge_index.to(device)
val_sparse_edge_index = val_sparse_edge_index.to(device)
val_exclude_edge_index = val_exclude_edge_index.to(device)
test_exclude_edge_index = test_exclude_edge_index.to(device)

# training loop
train_losses = []
val_losses = []

for iter in range(ITERATIONS):
    # forward propagation
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        train_sparse_edge_index)

    # mini batching
    user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
        BATCH_SIZE, train_edge_index)
    user_indices, pos_item_indices, neg_item_indices = user_indices.to(
        device), pos_item_indices.to(device), neg_item_indices.to(device)
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
        pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final = items_emb_final[neg_item_indices]
    neg_items_emb_0 = items_emb_0[neg_item_indices]

    # loss computation
    train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                          pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if iter % ITERS_PER_EVAL == 0:
        model.eval()
        val_loss, map_, ndcg, recall, precision = evaluation(
            model, val_edge_index, val_sparse_edge_index, [val_exclude_edge_index], TOPK, LAMBDA, validation_gt)
        print(f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}\n \
         val_map: {map_}\n val_ndcg: {ndcg}\n val_recall: {recall}\n val_precision: {precision}")
        train_losses.append(train_loss.item())
        val_losses.append(val_loss)
        model.train()

    if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
        scheduler.step()

    if iter % 500 == 0:
        save_ckpt(model, optimizer=optimizer, scheduler=scheduler, epoch=iter)        

iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]
plt.plot(iters, train_losses, label='train')
plt.plot(iters, val_losses, label='validation')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('training and validation loss curves')
plt.legend()
plt.savefig('../lightgcn_curve.png')

# evaluate on test set
model.eval()
test_edge_index = test_edge_index.to(device)
test_sparse_edge_index = test_sparse_edge_index.to(device)

test_loss, test_map, test_ndcg, test_recall, test_precision = evaluation(
            model, test_edge_index, test_sparse_edge_index, [val_exclude_edge_index, test_exclude_edge_index], TOPK, LAMBDA, test_gt)

print(f"[test_loss: {round(test_loss, 5)}, test_map: {test_map}\n \
 test_ndcg: {test_ndcg}\n test_recall: {test_recall}\n test_precision: {test_precision}")
