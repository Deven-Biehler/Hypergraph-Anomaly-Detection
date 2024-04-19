import matplotlib.pyplot as plt
import pandas as pd

def dataset_balance(data):
    label_1_count = data[data['Labels'] == 1].shape[0]
    label_0_count = data[data['Labels'] == 0].shape[0]
    print(f'Label 1 count: {label_1_count}')
    print(f'Label 0 count: {label_0_count}')
    print(f'Label 1 to Label 0 ratio: {label_1_count/label_0_count}')


data = pd.read_csv('data\S-FFSD\S-FFSD.csv')
dataset_balance(data)

