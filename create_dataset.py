from include.preprocess import *
import pickle
import json
import numpy as np
import hashlib

def create_data(size):
    # Create hypergraph edgelist
    write_hypergraph_edge_list("RC_2015-01", "hypergraph_edge_list", size)
    # Create labels
    get_labels("user_id_mapping", "labels")
    # Create train_mask
    get_train_mask("user_id_mapping", "train_mask")
    # Create val_mask
    get_val_mask("user_id_mapping", "val_mask")
    # Create test_mask
    get_test_mask("user_id_mapping", "test_mask")
    # Get text
    get_all_text("RC_2015-01", "corpus")
    # Create Features
    bow = bag_of_words("corpus", "features.pkl")


    edge_list = json.load(open("hypergraph_edge_list", "r"))
    with open("edgelist.pkl", "wb") as f:
        pickle.dump(edge_list, f)

    labels = json.load(open("labels", "r"))
    with open("labels.pkl", "wb") as f:
        labels = np.array(labels, dtype=np.int8)
        pickle.dump(labels, f)

    train_mask = json.load(open("train_mask", "r"))
    with open("train_mask.pkl", "wb") as f:
        pickle.dump(train_mask, f)

    val_mask = json.load(open("val_mask", "r"))
    with open("val_mask.pkl", "wb") as f:
        pickle.dump(val_mask, f)
    
    test_mask = json.load(open("test_mask", "r"))
    with open("test_mask.pkl", "wb") as f:
        pickle.dump(test_mask, f)


    num_classes = 2
    num_vertices = len(labels)
    num_edges = len(edge_list)
    dim_features = bow.shape[1]

    # Get MD5 for all pkl files
    md5_dict = {}
    file_list = ["edge_list.pkl", "features.pkl", "labels.pkl", "train_mask.pkl", "val_mask.pkl", "test_mask.pkl"]

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



    

if __name__ == "__main__":
    create_data(1000)