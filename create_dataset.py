from include.reddit.preprocess import *
import pickle
import json
import numpy as np
import hashlib

def create_data(size):
    # Create hypergraph edgelist
    write_hypergraph_edge_list("data/S-FFSD.csv", "data/hypergraph_edge_list", size)
    # Create labels
    get_labels("data/user_id_mapping", "data/labels")
    # Create train_mask
    get_train_mask("data/user_id_mapping", "data/train_mask")
    # Create val_mask
    get_val_mask("data/user_id_mapping", "data/val_mask")
    # Create test_mask
    get_test_mask("data/user_id_mapping", "data/test_mask")
    # Get text
    get_all_text("data/S-FFSD.csv", "data/corpus")
    # Create Features
    bow = bag_of_words("data/corpus", "data/pickles/features.pkl")


    edge_list = json.load(open("data/hypergraph_edge_list", "r"))
    with open("data/pickles/edgelist.pkl", "wb") as f:
        pickle.dump(edge_list, f)

    labels = json.load(open("data/labels", "r"))
    with open("data/pickles/labels.pkl", "wb") as f:
        labels = np.array(labels, dtype=np.int8)
        pickle.dump(labels, f)

    train_mask = json.load(open("data/train_mask", "r"))
    with open("data/pickles/train_mask.pkl", "wb") as f:
        pickle.dump(train_mask, f)

    val_mask = json.load(open("data/val_mask", "r"))
    with open("data/pickles/val_mask.pkl", "wb") as f:
        pickle.dump(val_mask, f)
    
    test_mask = json.load(open("data/test_mask", "r"))
    with open("data/pickles/test_mask.pkl", "wb") as f:
        pickle.dump(test_mask, f)


    num_classes = 2
    num_vertices = len(labels)
    num_edges = len(edge_list)
    dim_features = bow.shape[1]

    # Get MD5 for all pkl files
    md5_dict = {}
    file_list = ["data/pickles/edge_list.pkl", 
                 "data/pickles/features.pkl", 
                 "data/pickles/labels.pkl", 
                 "data/pickles/train_mask.pkl", 
                 "data/pickles/val_mask.pkl", 
                 "data/pickles/test_mask.pkl"]

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