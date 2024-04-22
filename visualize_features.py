import dhg
import numpy as np
import matplotlib.pyplot as plt
import dhg.visualization as vis
import json
import pickle
import pandas as pd

ft = open("data/Reddit/features.pkl", "rb")
ft = pickle.load(ft)
print(ft.shape)
ft = ft.toarray()
lbl = open("data/Reddit/labels.pkl", "rb")
lbl = pickle.load(lbl)
print(lbl.shape)
vis.draw_in_euclidean_space(ft, lbl)
plt.show()