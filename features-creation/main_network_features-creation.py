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

cloCentrality = nx.closeness_centrality(G)
df['user_closeness'] = df['user_id'].map(cloCentrality)
df['product_closeness'] = df['parent_asin'].map(cloCentrality)

eigCentrality = nx.eigenvector_centrality(G, max_iter=5000, tol=1e-06)
df['user_eig'] = df['user_id'].map(eigCentrality)
df['product_eig'] = df['parent_asin'].map(eigCentrality)

df.to_csv('../dataset/new_features.csv', index=False)