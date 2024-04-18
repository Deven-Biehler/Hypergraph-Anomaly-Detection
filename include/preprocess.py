import json
import networkx as nx
import time
import hypernetx as hnx
import json
import random
from sklearn.feature_extraction.text import CountVectorizer
import string
import pickle

def write_hypergraph_edge_list(input_file_path, output_file_path, size):
    write_file = open(output_file_path, "w")
    read_file = open(input_file_path, "r", encoding="utf-8")
    sucessful_entries = 0
    edgelist = {}
    user_ids = []
    for line in read_file:
        line_dict = json.loads(line)
        if line_dict["author"] == "[deleted]":
            continue
        if line_dict["author"] not in user_ids:
            user_ids.append(line_dict["author"])
        if line_dict["subreddit"] in edgelist:
            edgelist[line_dict["subreddit"]].append(line_dict["author"])
        else:
            edgelist[line_dict["subreddit"]] = [line_dict["author"]]
        sucessful_entries += 1
        if sucessful_entries >= size:
            break
    output = []
    for user_list in edgelist.values():
        output.append([user_ids.index(user) for user in user_list])
    json.dump(output, write_file)
    json.dump(user_ids, open("user_id_mapping", "w"), indent = 4)
        


def write_edge_list():
    timer = time.perf_counter()
    write_file = open("edge_list", "w")
    sucessful_entries = 0

    with open("RC_2015-01", "r") as file:
        line = file.readline()
        print(line)
        while line:
            line = file.readline()
            line_dict = json.loads(line)
            if line_dict["author"] == "[deleted]":
                continue
            write_file.write(line_dict["author"] + " " + line_dict["subreddit"] + "\n")
            sucessful_entries += 1
            if time.perf_counter() - timer > 360:
                print("Entries processed: " + str(sucessful_entries))
                break

    write_file.close()



def subset_edge_list(input_file_path, output_file_path, subset):
    edgelist_file_path = input_file_path
    edgelist_file = open(edgelist_file_path, "r")

    new_file = open(output_file_path, "w")
    for i, line in enumerate(edgelist_file):
        new_file.write(line)
        print(line)
        if i == subset:
            break
    new_file.close()

def get_labels(input_file_path, output_file_path):
    data = open(input_file_path, "r")
    data = json.load(data)

    # randomly select users for now
    labels = []
    for i, user in enumerate(data):
        if random.random() < 0.5:
            labels.append(1)
        else:
            labels.append(0)
    output_file = open(output_file_path, "w")
    json.dump(labels, output_file, indent = 4)

def get_train_mask(input_file_path, output_file_path):
    data = open(input_file_path, "r")
    data = json.load(data)

    # randomly select users for now
    train_mask = []
    for i, user in enumerate(data):
        if i % 6 == 0 or i % 6 == 1 or i % 6 == 2 or i % 6 == 3:
            train_mask.append(1)
        else:
            train_mask.append(0)
    output_file = open(output_file_path, "w")
    json.dump(train_mask, output_file, indent = 4)

def get_val_mask(input_file_path, output_file_path):
    data = open(input_file_path, "r")
    data = json.load(data)

    # randomly select users for now
    val_mask = []
    for i, user in enumerate(data):
        if i % 6 == 4:
            val_mask.append(1)
        else:
            val_mask.append(0)
    output_file = open(output_file_path, "w")
    json.dump(val_mask, output_file, indent = 4)

def get_test_mask(input_file_path, output_file_path):
    data = open(input_file_path, "r")
    data = json.load(data)

    # randomly select users for now
    test_mask = []
    for i, user in enumerate(data):
        if i % 6 == 5:
            test_mask.append(1)
        else:
            test_mask.append(0)
    output_file = open(output_file_path, "w")
    json.dump(test_mask, output_file, indent = 4)

    
def edgelist_to_hypergraph(input_file_path):
    hypergraph = {}
    with open(input_file_path, "r") as file:
        for line in file:
            subreddit, user = line.split(" ")
            if user in hypergraph:
                hypergraph[user].append(subreddit.strip())
            else:
                hypergraph[user] = [subreddit.strip()]
    output_file = open(input_file_path + "_hypergraph", "w")
    json.dump(hypergraph, output_file, indent = 4)

def read_hypergraph(input_file_path):
    with open(input_file_path, "r") as file:
        hypergraph = json.load(file)
    H = hnx.Hypergraph(hypergraph)
    return H
    
def gml_to_edgelist(input_file_path, output_file_path):
    G = nx.read_gml(input_file_path)
    nx.write_edgelist(G, output_file_path, delimiter=' ', data=False)

def get_users(input_file_path):
    users = set()
    with open(input_file_path, "r") as file:
        for line in file:
            users.add(line.split(" ")[1])
    file = open(input_file_path + "_" + "users", "w")
    for user in users:
        file.write(user)

def get_subreddits(input_file_path):
    subreddits = set()
    with open(input_file_path, "r") as file:
        for line in file:
            subreddits.add(line.split(" ")[0])
    file = open(input_file_path + "_" + "subreddits", "w")
    for subreddit in subreddits:
        file.write(subreddit + "\n")

def get_all_text(input_file_path, output_file_path):
    timer = time.perf_counter()
    write_file = open(output_file_path, "w", encoding="utf-8")
    sucessful_entries = 0

    with open(input_file_path, "r") as file:
        line = file.readline()
        print(line)
        user_text = {}
        while line:
            line = file.readline()
            line_dict = json.loads(line)
            if line_dict["author"] == "[deleted]":
                continue
            if line_dict["author"] in user_text:
                user_text[line_dict["author"]].append(clean_text(line_dict["body"]))
            else:
                user_text[line_dict["author"]] = [clean_text(line_dict["body"])]
            sucessful_entries += 1
            if sucessful_entries >= 1000:
                print("Entries processed: " + str(sucessful_entries))
                break
        json.dump(user_text, write_file)

    write_file.close()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def bag_of_words(input_file_path, output_file_path):
    data = open(input_file_path, "r", encoding="utf-8")
    data = json.load(data)

    vectorizer = CountVectorizer(ngram_range=(1,1))
    vectorizer.fit(data)
    
    bow = vectorizer.transform(data)
    
    with open(output_file_path, 'wb') as file:
        pickle.dump(bow, file)
    return bow

def generate_gml(input_file, output_gml_file, subsample_size):
    G = nx.bipartite.read_edgelist(input_file, create_using=nx.DiGraph)
    nx.write_gml(G, output_gml_file)