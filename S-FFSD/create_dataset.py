import pandas as pd
import networkx as nx
import dhg
import numpy as np
import pickle
import hashlib
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_file_path):
    data = pd.read_csv(input_file_path)
    data = data[data['Labels'] != 2]

    return data

def get_features(data):
    features = data.drop(columns=['Target', 'Labels'])
    features = np.array(features)
    label_encoder = LabelEncoder()
    for i in range(features.shape[1]):
        features[:, i] = label_encoder.fit_transform(features[:, i])
    features = np.array(features, dtype=np.float32)
    return features

def get_labels(data):
    labels = data['Labels'].values.tolist()
    return labels

def create_edgelist(data):
    hyper_edges = {}
    data["Transaction ID"] = range(len(data))
    for i, row in data.iterrows():
        if row['Target'] in hyper_edges.keys():
            hyper_edges[row['Target']].append(row['Transaction ID'])
        else:
            hyper_edges[row['Target']] = [row['Transaction ID']]
    return list(hyper_edges.values())

def get_train_mask(data):
    train_mask = []
    for i in range(len(data)):
        if i % 6 == 0 or i % 6 == 1 or i % 6 == 2 or i % 6 == 3:
            train_mask.append(1)
        else:
            train_mask.append(0)
    return train_mask

def get_val_mask(data):
    val_mask = []
    for i in range(len(data)):
        if i % 6 == 4:
            val_mask.append(1)
        else:
            val_mask.append(0)
    return val_mask

def get_test_mask(data):
    test_mask = []
    for i in range(len(data)):
        if i % 6 == 5:
            test_mask.append(1)
        else:
            test_mask.append(0)
    return test_mask



    
data = preprocess_data("data\S-FFSD\S-FFSD.csv")
edge_list = create_edgelist(data)
H = dhg.Hypergraph(len(data), edge_list)
features = get_features(data)
labels = get_labels(data)
train_mask = get_train_mask(data)
val_mask = get_val_mask(data)
test_mask = get_test_mask(data)


with open("data/S-FFSD/pickles/edgelist.pkl", "wb") as f:
    pickle.dump(edge_list, f)
with open("data/S-FFSD/pickles/features.pkl", "wb") as f:
    pickle.dump(features, f)
with open("data/S-FFSD/pickles/labels.pkl", "wb") as f:
    pickle.dump(labels, f)
with open("data/S-FFSD/pickles/train_mask.pkl", "wb") as f:
    pickle.dump(train_mask, f)
with open("data/S-FFSD/pickles/val_mask.pkl", "wb") as f:
    pickle.dump(val_mask, f)
with open("data/S-FFSD/pickles/test_mask.pkl", "wb") as f:
    pickle.dump(test_mask, f)

num_classes = 2
num_vertices = len(labels)
num_edges = len(edge_list)
dim_features = features.shape[1]

# Get MD5 for all pkl files
md5_dict = {}
file_list = ["data/S-FFSD/pickles/edgelist.pkl", 
                "data/S-FFSD/pickles/features.pkl", 
                "data/S-FFSD/pickles/labels.pkl", 
                "data/S-FFSD/pickles/train_mask.pkl", 
                "data/S-FFSD/pickles/val_mask.pkl", 
                "data/S-FFSD/pickles/test_mask.pkl"]

for file_name in file_list:
    with open(file_name, "rb") as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
        md5_dict[file_name] = md5_hash

print("MD5 hashes:")
for file_name, md5_hash in md5_dict.items():
    print(file_name, ":", md5_hash)


print("num_classes:", num_classes)
print("num_vertices:", num_vertices)
print("num_edges:", num_edges)
print("dim_features:", dim_features)
