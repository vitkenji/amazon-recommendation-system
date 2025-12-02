import pandas as pd
import networkx as nx
import numpy as np
import ast
from networkx.algorithms import bipartite
from sklearn.model_selection import train_test_split
import community as community_louvain

df = pd.read_csv('../dataset/subsample.csv')
df["categories"] = df["categories"].apply(ast.literal_eval)
df["number_categories"] = df["categories"].apply(len)
df = df.drop(columns=['rating', 'neg', 'neu', 'pos', 'compound', 'categories'], axis=1)
df["exists"] = 1

rng = np.random.default_rng(42)

pos_pairs = set(zip(df['user_id'], df['parent_asin']))

train_pos, test_pos = train_test_split(df, test_size=0.2, random_state=42)

product_info = df[['parent_asin', 'main_category', 'average_rating', 'rating_number', 'price', 'number_categories']].drop_duplicates()
product_dict = product_info.set_index('parent_asin').to_dict('index')
users_all = df['user_id'].unique()
products_all = product_info['parent_asin'].values

def sample_negatives(n, users, products, forbidden_pairs, product_dict, rng):
    neg = []
    existing = set(forbidden_pairs)
    while len(neg) < n:
        u = rng.choice(users)
        p = rng.choice(products)
        if (u, p) not in existing:
            pdict = product_dict[p]
            neg.append({
                'parent_asin': p,
                'user_id': u,
                'main_category': pdict['main_category'],
                'average_rating': pdict['average_rating'],
                'rating_number': pdict['rating_number'],
                'price': pdict['price'],
                'number_categories': pdict['number_categories'],
                'exists': 0
            })
            existing.add((u, p))
    return pd.DataFrame(neg)

train_neg = sample_negatives(len(train_pos), users_all, products_all, pos_pairs, product_dict, rng)
test_forbidden = pos_pairs.union(set(zip(train_neg['user_id'], train_neg['parent_asin'])))
test_neg = sample_negatives(len(test_pos), users_all, products_all, test_forbidden, product_dict, rng)

train_pos = train_pos.copy()
test_pos = test_pos.copy()
train_pos['exists'] = 1
test_pos['exists'] = 1

train_df = pd.concat([train_pos, train_neg], ignore_index=True)
test_df = pd.concat([test_pos, test_neg], ignore_index=True)

train_users = train_df['user_id'].unique()
train_products = train_df['parent_asin'].unique()

G_train = nx.Graph()
G_train.add_nodes_from(train_users, bipartite='users')
G_train.add_nodes_from(train_products, bipartite='products')
G_train.add_edges_from(list(zip(train_pos['user_id'], train_pos['parent_asin'])))

G_users = bipartite.projected_graph(G_train, train_users)
G_products = bipartite.projected_graph(G_train, train_products)

user_deg = dict(G_train.degree(train_users))
product_deg = dict(G_train.degree(train_products))

pagerank = nx.pagerank(G_train)
user_pagerank = {n: pagerank.get(n, 0) for n in train_users}
product_pagerank = {n: pagerank.get(n, 0) for n in train_products}

try:
    user_closeness = nx.closeness_centrality(G_train.subgraph(train_users))
except Exception:
    user_closeness = {u: 0 for u in train_users}
try:
    product_closeness = nx.closeness_centrality(G_train.subgraph(train_products))
except Exception:
    product_closeness = {p: 0 for p in train_products}

try:
    eig = nx.eigenvector_centrality(G_train, max_iter=5000, tol=1e-06)
except Exception:
    eig = {}
user_eig = {n: eig.get(n, 0) for n in train_users}
product_eig = {n: eig.get(n, 0) for n in train_products}

c_p = {}
if len(G_products) > 0:
    c_p = community_louvain.best_partition(G_products)
c_u = {}
if len(G_users) > 0:
    c_u = community_louvain.best_partition(G_users)

def map_feature(df_in):
    df_local = df_in.copy()
    df_local['user_deg'] = df_local['user_id'].map(user_deg).fillna(0)
    df_local['product_deg'] = df_local['parent_asin'].map(product_deg).fillna(0)
    df_local['user_pagerank'] = df_local['user_id'].map(user_pagerank).fillna(0)
    df_local['product_pagerank'] = df_local['parent_asin'].map(product_pagerank).fillna(0)
    df_local['user_closeness'] = df_local['user_id'].map(user_closeness).fillna(0)
    df_local['product_closeness'] = df_local['parent_asin'].map(product_closeness).fillna(0)
    df_local['user_eig'] = df_local['user_id'].map(user_eig).fillna(0)
    df_local['product_eig'] = df_local['parent_asin'].map(product_eig).fillna(0)
    df_local['product_community'] = df_local['parent_asin'].map(c_p).astype('category').cat.add_categories([-1]).fillna(-1)
    df_local['user_community'] = df_local['user_id'].map(c_u).astype('category').cat.add_categories([-1]).fillna(-1)
    return df_local

train_df = map_feature(train_df)
test_df = map_feature(test_df)

def common_neighbors_user_side(u, p):
    if u not in G_users or p not in G_train:
        return 0
    users_who_rated_p = set(G_train.neighbors(p)) if p in G_train else set()
    neighbors_u = set(G_users.neighbors(u)) if u in G_users else set()
    return len(users_who_rated_p & neighbors_u)

def common_neighbors_product_side(u, p):
    if p not in G_products or u not in G_train:
        return 0
    products_rated_by_u = set(G_train.neighbors(u)) if u in G_train else set()
    neighbors_p = set(G_products.neighbors(p)) if p in G_products else set()
    return len(products_rated_by_u & neighbors_p)

def jaccard_user_side(u, p):
    if p not in G_train or u not in G_users:
        return 0
    users_who_rated_p = set(G_train.neighbors(p))
    neighbors_u = set(G_users.neighbors(u))
    if not users_who_rated_p or not neighbors_u:
        return 0
    return len(users_who_rated_p & neighbors_u) / len(users_who_rated_p | neighbors_u)

def jaccard_product_side(u, p):
    if u not in G_train or p not in G_products:
        return 0
    products_rated_by_u = set(G_train.neighbors(u))
    neighbors_p = set(G_products.neighbors(p))
    if not products_rated_by_u or not neighbors_p:
        return 0
    return len(products_rated_by_u & neighbors_p) / len(products_rated_by_u | neighbors_p)

def adamic_adar_product_side(u, p):
    if u not in G_train or p not in G_products:
        return 0
    products_rated_by_u = set(G_train.neighbors(u))
    inter = products_rated_by_u & set(G_products.neighbors(p))
    scores = 0
    for z in inter:
        deg = len(list(G_products.neighbors(z)))
        if deg > 1:
            scores += 1 / np.log(deg)
    return scores

def preferential_attachment(u, p):
    du = G_train.degree(u) if u in G_train else 0
    dp = G_train.degree(p) if p in G_train else 0
    return du * dp

train_df["preferential_attachment"] = train_df.apply(lambda row: preferential_attachment(row["user_id"], row["parent_asin"]), axis=1)
train_df["common_neighbors_product"] = train_df.apply(lambda row: common_neighbors_product_side(row["user_id"], row["parent_asin"]), axis=1)
train_df["jaccard_product"] = train_df.apply(lambda row: jaccard_product_side(row["user_id"], row["parent_asin"]), axis=1)
train_df["adamic_adar_product"] = train_df.apply(lambda row: adamic_adar_product_side(row["user_id"], row["parent_asin"]), axis=1)
train_df["common_neighbors_user"] = train_df.apply(lambda row: common_neighbors_user_side(row["user_id"], row["parent_asin"]), axis=1)
train_df["jaccard_user"] = train_df.apply(lambda row: jaccard_user_side(row["user_id"], row["parent_asin"]), axis=1)

test_df["preferential_attachment"] = test_df.apply(lambda row: preferential_attachment(row["user_id"], row["parent_asin"]), axis=1)
test_df["common_neighbors_product"] = test_df.apply(lambda row: common_neighbors_product_side(row["user_id"], row["parent_asin"]), axis=1)
test_df["jaccard_product"] = test_df.apply(lambda row: jaccard_product_side(row["user_id"], row["parent_asin"]), axis=1)
test_df["adamic_adar_product"] = test_df.apply(lambda row: adamic_adar_product_side(row["user_id"], row["parent_asin"]), axis=1)
test_df["common_neighbors_user"] = test_df.apply(lambda row: common_neighbors_user_side(row["user_id"], row["parent_asin"]), axis=1)
test_df["jaccard_user"] = test_df.apply(lambda row: jaccard_user_side(row["user_id"], row["parent_asin"]), axis=1)

train_df['deg_ratio'] = train_df['product_deg'] / (train_df['user_deg'] + 1e-10)
train_df['pagerank_ratio'] = train_df['product_pagerank'] / (train_df['user_pagerank'] + 1e-10)
train_df['closeness_diff'] = abs(train_df['product_closeness'] - train_df['user_closeness'])
train_df['eig_ratio'] = train_df['product_eig'] / (train_df['user_eig'] + 1e-10)
train_df['product_deg_zscore'] = (train_df['product_deg'] - train_df['product_deg'].mean()) / (train_df['product_deg'].std() + 1e-10)
train_df['rating_number_log'] = np.log1p(train_df['rating_number'])
train_df['high_rating_high_reviews'] = ((train_df['average_rating'] > 4.5) & (train_df['rating_number'] > train_df['rating_number'].median())).astype(int)
train_df['active_user'] = (train_df['user_deg'] > train_df['user_deg'].quantile(0.75)).astype(int)
train_df['user_centrality_score'] = (train_df['user_pagerank'] * train_df['user_closeness'] * train_df['user_eig']) ** (1/3)

test_df['deg_ratio'] = test_df['product_deg'] / (test_df['user_deg'] + 1e-10)
test_df['pagerank_ratio'] = test_df['product_pagerank'] / (test_df['user_pagerank'] + 1e-10)
test_df['closeness_diff'] = abs(test_df['product_closeness'] - test_df['user_closeness'])
test_df['eig_ratio'] = test_df['product_eig'] / (test_df['user_eig'] + 1e-10)
test_df['product_deg_zscore'] = (test_df['product_deg'] - train_df['product_deg'].mean()) / (train_df['product_deg'].std() + 1e-10)
test_df['rating_number_log'] = np.log1p(test_df['rating_number'])
test_df['high_rating_high_reviews'] = ((test_df['average_rating'] > 4.5) & (test_df['rating_number'] > train_df['rating_number'].median())).astype(int)
test_df['active_user'] = (test_df['user_deg'] > train_df['user_deg'].quantile(0.75)).astype(int)
test_df['user_centrality_score'] = (test_df['user_pagerank'] * test_df['user_closeness'] * test_df['user_eig']) ** (1/3)

cols_to_encode = ['product_community', 'user_community', 'main_category']
combined_for_encoding = pd.concat([train_df, test_df], axis=0)
combined_for_encoding = pd.get_dummies(combined_for_encoding, columns=cols_to_encode, prefix=cols_to_encode)
n_train = len(train_df)
train_df = combined_for_encoding.iloc[:n_train].copy()
test_df = combined_for_encoding.iloc[n_train:].copy()

train_df.to_csv('../dataset/train_df_v2.csv', index=False)
test_df.to_csv('../dataset/test_df_v2.csv', index=False)