import pandas as pd
import networkx as nx
from nltk.sentiment import SentimentIntensityAnalyzer

# since initially there was 25Gb of data, the code below just demonstrates how pre-processing was executed
# products.csv and reviews.csv were deleted.
# dataset now is 100Mb.

# cleans products dataset, removing duplicates, rows with null values etc.

products = pd.read_csv('../data/products.csv')
products = products.drop_duplicates()
products = products.dropna()
products['main_category'] = products['main_category'].str.lower()
products['main_category'] = products['main_category'].str.strip()
products.to_csv('../data/products.csv', index=False)

# cleans reviews dataset, removing duplicates, rows with null values etc.
reviews = pd.read_csv('../data/reviews.csv')
reviews = reviews.drop_duplicates()
reviews = reviews.dropna()

# merges data and cleans it

df = pd.merge(reviews,products,on='parent_asin',how='left')
df = df.drop_duplicates()
df = df.dropna()

# selects most important nodes
user_counts = df.groupby('user_id')['user_id'].transform('count')
product_counts = df.groupby('parent_asin')['parent_asin'].transform('count')

while True:
    initial_rows = len(df)

    df = df[df.groupby('parent_asin')['parent_asin'].transform('size') >= 40]
    
    df = df[df.groupby('user_id')['user_id'].transform('size') >= 8]

    if len(df) == initial_rows:
        break
        
# extract sentiment of text
sent = SentimentIntensityAnalyzer()
df[['neg', 'neu', 'pos', 'compound']] = df['text'].apply(lambda x: pd.Series(sent.polarity_scores(x)))
df = df.drop(columns=['text'], axis=1)

df = df.drop_duplicates()
df = df.dropna()

df.to_csv('../data/final_data.csv', index=False)

# creates network

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

nx.write_gml(G, "../reviews_network.gml")