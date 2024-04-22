import networkx as nx
import pandas as pd
from pandas import read_csv

def create_bipartite_graph(data):
    G = nx.Graph()
    for i, row in data.iterrows():
        G.add_node(row['Source'], bipartite=0)
        G.add_node(row['Target'], bipartite=1)
        G.add_edge(row['Source'], row['Target'])
    return G
data = read_csv('data\S-FFSD\S-FFSD.csv')
G = create_bipartite_graph(data)
nx.write_gml(G, 'graph.gml')