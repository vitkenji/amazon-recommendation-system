import os
import pandas as pd
import numpy as np

"""
Gera um arquivo de features enriquecidas a partir de new_features.csv.

Adições principais:
- Produto: log_rating_number, log_price, quality_pop, prod_compound_mean
- Usuário: user_avg_price, user_price_std, user_compound_mean
- Par (user, product): price_diff, price_z (se std>0), has_main_category_before, same_main_category

Saída: ../dataset/enriched_features.csv
"""

CONFIG = {
    'INPUT_FEATURES_PATH': '../dataset/new_features.csv',
    'OUTPUT_ENRICHED_PATH': '../dataset/enriched_features.csv',
}

def main():
    in_path = CONFIG['INPUT_FEATURES_PATH']
    out_path = CONFIG['OUTPUT_ENRICHED_PATH']
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {in_path}")

    df = pd.read_csv(in_path)
    # Remover coluna estranha se presente
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Garantir colunas necessárias
    required_cols = ['user_id', 'parent_asin', 'average_rating', 'rating_number', 'price', 'compound', 'main_category',
                     'user_deg', 'product_deg', 'user_pagerank', 'product_pagerank',
                     'user_closeness', 'product_closeness', 'user_eig', 'product_eig',
                     'user_community', 'product_community']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente em new_features.csv: {c}")

    # Agregações por produto
    prod_agg = df.groupby('parent_asin').agg(
        average_rating=('average_rating', 'first'),
        rating_number=('rating_number', 'first'),
        price=('price', 'first'),
        prod_compound_mean=('compound', 'mean'),
        main_category=('main_category', 'first'),
    ).reset_index().rename(columns={'parent_asin': 'product_id'})
    prod_agg['log_rating_number'] = np.log1p(prod_agg['rating_number'].fillna(0))
    prod_agg['log_price'] = np.log1p(prod_agg['price'].fillna(0))
    prod_agg['quality_pop'] = prod_agg['average_rating'].fillna(0) * prod_agg['log_rating_number']

    # Agregações por usuário
    user_agg = df.groupby('user_id').agg(
        user_avg_price=('price', 'mean'),
        user_price_std=('price', 'std'),
        user_compound_mean=('compound', 'mean'),
    ).reset_index()
    user_agg['user_price_std'] = user_agg['user_price_std'].fillna(0.0)

    # Mapas auxiliares
    user_cat_set = df.groupby('user_id')['main_category'].apply(lambda s: set(s.dropna().astype(str).tolist()))
    prod_cat_map = prod_agg.set_index('product_id')['main_category'].astype(str).to_dict()

    # Juntar de volta por par (user, product)
    base_pairs = df[['user_id', 'parent_asin',
                     'user_deg', 'product_deg', 'user_pagerank', 'product_pagerank',
                     'user_closeness', 'product_closeness', 'user_eig', 'product_eig',
                     'user_community', 'product_community']].drop_duplicates()

    # Anexar agregações
    enriched = base_pairs.merge(user_agg, on='user_id', how='left')
    enriched = enriched.merge(prod_agg, left_on='parent_asin', right_on='product_id', how='left')

    # Features derivadas par
    # price_diff: diferença absoluta entre preço do produto e média do usuário
    enriched['price_diff'] = (enriched['price'] - enriched['user_avg_price']).abs()

    # price_z: (price - user_avg_price) / user_price_std se std>0
    std_nonzero = enriched['user_price_std'].replace(0, np.nan)
    enriched['price_z'] = (enriched['price'] - enriched['user_avg_price']) / std_nonzero
    enriched['price_z'] = enriched['price_z'].fillna(0.0)

    # same_main_category: produto tem mesma categoria principal de algum histórico do usuário?
    def has_cat_before(u, p):
        cat = prod_cat_map.get(p)
        uset = user_cat_set.get(u, set())
        return 1 if (cat in uset) else 0

    enriched['has_main_category_before'] = [has_cat_before(u, p) for u, p in zip(enriched['user_id'], enriched['parent_asin'])]
    enriched['same_main_category'] = enriched['has_main_category_before']

    # Seleção final de colunas (mantemos todas necessárias ao treino/predição)
    # Observação: mantemos product_id, main_category e agregados para possível uso futuro
    cols_order = [
        'user_id', 'parent_asin',
        # grafo base
        'user_deg', 'product_deg', 'user_pagerank', 'product_pagerank',
        'user_closeness', 'product_closeness', 'user_eig', 'product_eig',
        'user_community', 'product_community',
        # produto agregados
        'average_rating', 'rating_number', 'log_rating_number', 'price', 'log_price', 'quality_pop',
        'prod_compound_mean', 'main_category',
        # usuário agregados
        'user_avg_price', 'user_price_std', 'user_compound_mean',
        # par derivadas
        'price_diff', 'price_z', 'has_main_category_before', 'same_main_category',
    ]
    enriched = enriched[cols_order]

    # Salvar
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"Features enriquecidas salvas em: {out_path}")

if __name__ == '__main__':
    main()
