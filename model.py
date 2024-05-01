import time
from copy import deepcopy

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from include.reddit.reddit_graph_dataset import Reddit
from dhg.data import Facebook, Cooking200
from dhg.models import HGNNP
from dhg.random import set_seed
from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator
from sklearn.metrics import roc_auc_score

# define your train function
def train(net, X, A, lbls, train_idx, optimizer, epoch, loss_list, accuracy_list):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    preds = torch.argmax(outs, dim=1)
    accuracy = torch.sum(preds == lbls).item() / len(lbls)
    num_zeros = torch.sum(preds == 0).item()
    num_ones = torch.sum(preds == 1).item()
    print(f"Number of 0s in preds: {num_zeros}")
    print(f"Number of 1s in preds: {num_ones}")
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}, Accuracy: {accuracy:.5f}")
    
    # Save loss and accuracy
    loss_list.append(loss.item())
    accuracy_list.append(accuracy)
    
    return loss.item()

@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)

    auc_score = roc_auc_score(lbls.cpu().numpy(), outs.cpu().numpy()[:, 1])
    print(f"AUC Score: {auc_score:.5f}")
    
    return res

if __name__ == "__main__":
    set_seed(2022)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # config your evaluation metric here
    evaluator = Evaluator(["accuracy", "f1_score", "confusion_matrix", {"f1_score": {"average": "micro"}}])
    # load Your data here
    data = Reddit()
    print(data.raw("features").shape)
    X, lbl = data["features"], data["labels"]
    # construct your correlation structure here
    G = Hypergraph(data["num_vertices"], data["edge_list"])
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    # initialize your model here
    net = HGNNP(data["dim_features"], 16, data["num_classes"])
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    loss_list = []
    accuracy_list = []
    for epoch in range(250):
        # train
        train_loss = train(net, X, G, lbl, train_mask, optimizer, epoch, loss_list, accuracy_list)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, G, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # testing
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, G, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
    
    # Plot loss and accuracy
    plt.plot(loss_list, label='Loss')
    plt.plot(accuracy_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()