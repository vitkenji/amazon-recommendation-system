import pandas as pd
import networkx as nx
import networkit as nk
import json

df = pd.read_csv('../dataset/subsample.csv')
G = nx.read_gml('../network/reviews_network.gml')

nkG = nk.nxadapter.nx2nk(G)

deg = nk.centrality.DegreeCentrality(nkG, normalized=True)
deg.run()
degreeCentrality = dict(zip(G.nodes(), deg.scores()))

df['user_deg'] = df['user_id'].map(degreeCentrality)
df['product_deg'] = df['parent_asin'].map(degreeCentrality)

pageRank = nx.pagerank(G)
df['user_pagerank'] = df['user_id'].map(pageRank)
df['product_pagerank'] = df['parent_asin'].map(pageRank)
