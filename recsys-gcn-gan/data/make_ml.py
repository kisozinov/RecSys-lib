import pandas as pd
from torch_geometric.data import extract_zip
from sklearn.model_selection import train_test_split
from typing import Dict
from amazmemllib.preprocessing import LabelEncoder, LabelEncodingRule
#extract_zip('../data/raw/ml-1m.zip', '../data/raw')

rating_path = '../data/raw/ml-1m/ratings.dat'

df = pd.read_csv(rating_path, sep='::', usecols=[0,1], names=['user_id', 'item_id'])

encoder = LabelEncoder(
    [LabelEncodingRule("user_id"), LabelEncodingRule("item_id")]
)
transformed_data = encoder.fit_transform(df)
#print(transformed_data)

train, test = train_test_split(transformed_data, test_size=0.2)

def get_positives(data):
    edge_index = []
    edge_index.append(data['user_id'].to_list())
    edge_index.append(data['item_id'].to_list())

    user_pos_items = {}
    for i in range(len(edge_index[0])):
        user = edge_index[0][i]
        item = edge_index[1][i]
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items

train_dict = get_positives(train)
test_dict = get_positives(test)

def dict2txt(data_dict: Dict, mode: str):
    with open(f'../data/raw/ml-1m/{mode}.txt', 'w') as f:
        for user, items in data_dict.items():
            #print(list(map(str, [user]+items)))
            f.write(' '.join(map(str, [user]+items))+'\n')

dict2txt(train_dict, 'train')
dict2txt(test_dict, 'test')
