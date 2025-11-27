import pandas as pd
import networkx as nx

df = pd.read_csv('../dataset/subsample.csv')
G = nx.read_gml('../network/reviews_network.gml')

degreeCentrality = nx.degree_centrality(G)
df['user_deg'] = df['user_id'].map(degreeCentrality)
df['product_deg'] = df['parent_asin'].map(degreeCentrality)

pageRank = nx.pagerank(G)
df['user_pagerank'] = df['user_id'].map(pageRank)
df['product_pagerank'] = df['parent_asin'].map(pageRank)

df.to_csv('../dataset/new_features.csv', index=False)
