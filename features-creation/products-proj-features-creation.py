import collections
import pandas as pd
import networkx as nx
import community as community_louvain
from collections import Counter, defaultdict

G = nx.read_gml('../network/products.gml')
df = pd.read_csv('../dataset/new_features.csv')

aa_list = []
for u, v, score in nx.adamic_adar_index(G):
    aa_list.append((u, v, score))

adamic_adar_df = pd.DataFrame(aa_list, columns=['product_u', 'product_v', 'AA_score'])

jc_list = []
for u, v, score in nx.jaccard_coefficient(G):
    jc_list.append((u, v, score))

jaccard_df = pd.DataFrame(jc_list, columns=['product_u', 'product_v', 'Jaccard'])

c = community_louvain.best_partition(G)
df['product_community'] = df['parent_asin'].map(c)