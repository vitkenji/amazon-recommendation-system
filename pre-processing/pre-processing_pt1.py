import pandas as pd
import os
import networkx as nx
from nltk.sentiment import SentimentIntensityAnalyzer
from networkx.algorithms import bipartite

# loads reviews dataset
os.makedirs("../dataset/reviews_chunks", exist_ok=True)
reviews_files = []
for i, chunk in enumerate(pd.read_json( "../dataset/reviews.jsonl",lines=True, chunksize=100_000)):
    out_path = os.path.join("../dataset/reviews_chunks", f"reviews_chunk_{i}.parquet")
    chunk.to_parquet(out_path)
    reviews_files.append(out_path)

# cleans reviews dataset
reviews = pd.concat([pd.read_parquet(p) for p in reviews_files], ignore_index=True)
reviews = reviews.drop(columns=['title', 'images', 'asin', 'timestamp', 'helpful_vote', 'verified_purchase'], axis=1)
reviews = reviews.drop_duplicates()
reviews = reviews.dropna()

# loads products dataset
chunks = []
for chunk in pd.read_json("../dataset/products.jsonl", lines=True, chunksize=100_000):
    chunks.append(chunk)

# cleans products dataset
products = pd.concat(chunks, ignore_index=True)
products = products.drop(columns=['title', 'features', 'description', 'images', 'videos', 'store', 'details', 'bought_together', 'subtitle', 'author'], axis=1)
products = products.drop_duplicates()
products = products.dropna()
products['main_category'] = products['main_category'].str.lower()
products['main_category'] = products['main_category'].str.strip()
products['categories'] = products['categories'].str.lower()
products['categories'] = products['categories'].str.strip()

# filters reviews
user_counts = reviews.groupby("user_id")["user_id"].transform("size")
product_counts = reviews.groupby("parent_asin")["parent_asin"].transform("size")
reviews_filtered = reviews[(user_counts >= 60) & (product_counts >= 1200)]

# merges data and cleans it
df = pd.merge(reviews_filtered,products,on='parent_asin',how='left')
df = df.drop_duplicates()
df = df.dropna()

# extract sentiment of text
sent = SentimentIntensityAnalyzer()
df[['neg', 'neu', 'pos', 'compound']] = df['text'].apply(lambda x: pd.Series(sent.polarity_scores(x)))
df = df.drop(columns=['text'], axis=1)

# creates main network
G = nx.Graph()

for _, row in df.drop_duplicates(subset='parent_asin').iterrows():
    G.add_node(
        row['parent_asin'],
        type='product',
        main_category=row['main_category'],
        average_rating=row['average_rating'],
        rating_number=row['rating_number'],
        price=row['price']
    )

for user in df['user_id'].unique():
    G.add_node(user, type='user')

for _, row in df.iterrows():
    G.add_edge(
        row['user_id'],
        row['parent_asin'],
        rating=row['rating'],
        neg=row['neg'],
        neu=row['neu'],
        pos=row['pos'],
        compound=row['compound']
    )

# creates projections

users = set()
products = set()

for node, data in G.nodes(data=True):
    if data.get('type') == 'user':
        users.add(node)
    elif data.get('type') == 'product':
        products.add(node)

G_products = bipartite.projected_graph(G, nodes=products)

# exports data
df.to_csv('../dataset/subsample.csv', index=False)
nx.write_gml(G, "../network/reviews_network.gml")
nx.write_gml(G_users, '../network/users.gml')
nx.write_gml(G_products, '../network/products.gml')
 