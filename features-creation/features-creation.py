import pandas as pd
import networkx as nx

df = pd.read_csv('../dataset/preprocessed_data.csv')
G = nx.read_gml('../network/reviews_network.gml')

degreeCentrality = nx.degree_centrality(G)
betCentrality = nx.betweenness_centrality(G)
cloCentrality = nx.closeness_centrality(G)
eigCentrality = nx.eigenvector_centrality(G)
pageRank = nx.pagerank(G)

df['user_deg'] = df['user_id'].map(degreeCentrality)
df['product_deg'] = df['parent_asin'].map(degreeCentrality)
df['user_bet'] = df['user_id'].map(betCentrality)
df['product_bet'] = df['parent_asin'].map(betCentrality)
df['user_clo'] = df['user_id'].map(cloCentrality)
df['product_clo'] = df['parent_asin'].map(cloCentrality)
df['user_eig'] = df['user_id'].map(eigCentrality)
df['product_eig'] = df['parent_asin'].map(eigCentrality)
df['user_pagerank'] = df['user_id'].map(pageRankCentrality)
df['product_pagerank'] = df['parent_asin'].map(pageRankCentrality)