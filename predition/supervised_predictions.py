import os
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
"""

Etapas:
1. Carrega new_features.csv e hidden_links.csv (positivos de teste).
2. Constrói conjunto de treino (positivos = new_features - hidden_links).
3. Amostra negativos para treino por usuário.
4. Cria features numéricas + same_community.
5. Treina modelo (LogisticRegression).
6. Gera candidatos de teste por usuário (inclui todos hidden + amostra de produtos adicionais).
7. Prediz probabilidade do link e monta recommendations_topk_supervised.csv

"""

# -------------------------------------------------------------
# CONFIGURAÇÃO (edite aqui)
# -------------------------------------------------------------
CONFIG = {
    'NEW_FEATURES_PATH': '../dataset/new_features.csv',  # arquivo com features dos links observados
    'HIDDEN_LINKS_PATH': '../dataset/hidden_links.csv',  # gerado pelo build_training_graph.py
    'OUTPUT_SUPERVISED_PATH': '../dataset/recommendations_topk_supervised.csv',
    'OUTPUT_STANDARD_PATH': '../dataset/recommendations_topk.csv',  # arquivo padrão consumido pelo avaliador
    'TOP_K': 10,                      # quantidade de recomendações por usuário
    'FRACTION_TRAIN': 1.0,            # fração dos positivos de treino a usar (amostra para acelerar)
    'NEG_TRAIN_PER_POS': 2,           # negativos de treino por positivo
    'TEST_EXTRA_CANDIDATES_PER_USER': 100,  # nº extra de produtos (além dos hidden) para scoring no teste por usuário
    'RANDOM_SEED': 42,
    'USE_STANDARD_FILE': True,        # se True, salva também no nome padrão para avaliação reutilizando script atual
}

# Se OUTPUT_TAG for usado no pipeline, aplicamos sufixos aos caminhos.
OUTPUT_TAG = os.environ.get('OUTPUT_TAG')
if OUTPUT_TAG:
    def suffixed(path):
        base, ext = os.path.splitext(os.path.basename(path))
        return os.path.join(os.path.dirname(path), f"{base}__{OUTPUT_TAG}{ext}")
    CONFIG['HIDDEN_LINKS_PATH'] = suffixed(CONFIG['HIDDEN_LINKS_PATH'])
    CONFIG['OUTPUT_SUPERVISED_PATH'] = suffixed(CONFIG['OUTPUT_SUPERVISED_PATH'])
    CONFIG['OUTPUT_STANDARD_PATH'] = suffixed(CONFIG['OUTPUT_STANDARD_PATH'])

# Overrides via variáveis de ambiente quando chamado pelo pipeline
env_top_k = os.environ.get('SUP_TOP_K')
env_neg_train = os.environ.get('SUP_NEG_TRAIN_PER_POS')
env_test_extra = os.environ.get('SUP_TEST_EXTRA')
env_fraction_train = os.environ.get('SUP_FRACTION_TRAIN')
env_use_standard = os.environ.get('SUP_USE_STANDARD_FILE')

if env_top_k is not None:
    CONFIG['TOP_K'] = int(float(env_top_k))
if env_neg_train is not None:
    CONFIG['NEG_TRAIN_PER_POS'] = int(float(env_neg_train))
if env_test_extra is not None:
    CONFIG['TEST_EXTRA_CANDIDATES_PER_USER'] = int(float(env_test_extra))
if env_fraction_train is not None:
    CONFIG['FRACTION_TRAIN'] = float(env_fraction_train)
if env_use_standard is not None:
    CONFIG['USE_STANDARD_FILE'] = (env_use_standard.strip() in ['1', 'true', 'True'])

random.seed(CONFIG['RANDOM_SEED'])
np.random.seed(CONFIG['RANDOM_SEED'])

# -------------------------------------------------------------
# 1. Carregar dados
# -------------------------------------------------------------
if not os.path.exists(CONFIG['NEW_FEATURES_PATH']):
    raise FileNotFoundError(f"NEW_FEATURES_PATH não encontrado: {CONFIG['NEW_FEATURES_PATH']}")
if not os.path.exists(CONFIG['HIDDEN_LINKS_PATH']):
    raise FileNotFoundError(f"HIDDEN_LINKS_PATH não encontrado: {CONFIG['HIDDEN_LINKS_PATH']}")

all_links_df = pd.read_csv(CONFIG['NEW_FEATURES_PATH'])
# Normalizar nomes potencialmente estranhos
if 'Unnamed: 0' in all_links_df.columns:
    all_links_df = all_links_df.drop(columns=['Unnamed: 0'])

hidden_df = pd.read_csv(CONFIG['HIDDEN_LINKS_PATH'])  # colunas: user_id, parent_asin
hidden_df = hidden_df[['user_id', 'parent_asin']].drop_duplicates()

# -------------------------------------------------------------
# 2. Construir conjunto de treino (positivos = all_links - hidden)
# -------------------------------------------------------------
# Anti-join: remover pares que estão em hidden_df
hidden_pairs = set(map(tuple, hidden_df[['user_id', 'parent_asin']].values))
train_pos_df = all_links_df[~all_links_df[['user_id', 'parent_asin']].apply(tuple, axis=1).isin(hidden_pairs)].copy()
train_pos_df = train_pos_df[['user_id', 'parent_asin', 'user_deg', 'product_deg',
                             'user_pagerank', 'product_pagerank', 'user_closeness', 'product_closeness',
                             'user_eig', 'product_eig', 'user_community', 'product_community']]

if CONFIG['FRACTION_TRAIN'] < 1.0:
    train_pos_df = train_pos_df.sample(frac=CONFIG['FRACTION_TRAIN'], random_state=CONFIG['RANDOM_SEED'])

print(f"Positivos de treino: {len(train_pos_df)}")
print(f"Positivos de teste (hidden): {len(hidden_df)}")

# -------------------------------------------------------------
# 3. Preparar features base para síntese de negativos
# -------------------------------------------------------------
# Para negativos, precisamos das features do usuário e do produto separadas.
user_cols = ['user_deg', 'user_pagerank', 'user_closeness', 'user_eig', 'user_community']
prod_cols = ['product_deg', 'product_pagerank', 'product_closeness', 'product_eig', 'product_community']

# Pegar primeira ocorrência por user / product
user_feat_df = train_pos_df.groupby('user_id')[user_cols].first().reset_index()
prod_feat_df = train_pos_df.groupby('parent_asin')[prod_cols].first().reset_index().rename(columns={'parent_asin': 'product_id'})

# Conjunto de produtos disponíveis para sampling
available_products = set(prod_feat_df['product_id'])

# Map rápido para usuário -> produtos positivos de treino
user_train_pos_products = train_pos_df.groupby('user_id')['parent_asin'].apply(set).to_dict()

# Map rápido para usuário -> produtos escondidos (hidden) para evitar filtrar a cada iteração
hidden_map = hidden_df.groupby('user_id')['parent_asin'].apply(set).to_dict()

# -------------------------------------------------------------
# 4. Gerar negativos de treino
# -------------------------------------------------------------
neg_rows = []
for user, pos_set in user_train_pos_products.items():
    # Produtos candidatos para negativos: disponíveis menos os já positivos e menos hidden do usuário
    hidden_for_user = hidden_map.get(user, set())
    candidates = list(available_products - pos_set - hidden_for_user)
    if not candidates:
        continue
    # Quantidade alvo de negativos para este usuário
    target_neg = len(pos_set) * CONFIG['NEG_TRAIN_PER_POS']
    sampled = random.sample(candidates, min(target_neg, len(candidates)))
    # Obter features do usuário
    urow = user_feat_df[user_feat_df['user_id'] == user].iloc[0]
    for prod in sampled:
        prow = prod_feat_df[prod_feat_df['product_id'] == prod].iloc[0]
        neg_rows.append({
            'user_id': user,
            'parent_asin': prod,
            'user_deg': urow['user_deg'],
            'product_deg': prow['product_deg'],
            'user_pagerank': urow['user_pagerank'],
            'product_pagerank': prow['product_pagerank'],
            'user_closeness': urow['user_closeness'],
            'product_closeness': prow['product_closeness'],
            'user_eig': urow['user_eig'],
            'product_eig': prow['product_eig'],
            'user_community': urow['user_community'],
            'product_community': prow['product_community'],
            'label': 0,
        })

train_neg_df = pd.DataFrame(neg_rows)
print(f"Negativos de treino gerados: {len(train_neg_df)}")

# -------------------------------------------------------------
# 5. Marcar positivos de treino com label=1 e concatenar
# -------------------------------------------------------------
train_pos_df = train_pos_df.copy()
train_pos_df['label'] = 1
train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True)

# Feature same_community
train_df['same_community'] = (train_df['user_community'] == train_df['product_community']).astype(int)

# Feature de interação: preferential attachment (produto dos graus)
train_df['deg_pa'] = train_df['user_deg'] * train_df['product_deg']

# -------------------------------------------------------------
# 6. Selecionar colunas de features
# -------------------------------------------------------------
FEATURE_COLS = [
    'user_deg', 'product_deg',
    'user_pagerank', 'product_pagerank',
    'user_closeness', 'product_closeness',
    'user_eig', 'product_eig',
    'same_community',
    'deg_pa'
]
# (Para mudar features, edite FEATURE_COLS.)

X_train = train_df[FEATURE_COLS].values
y_train = train_df['label'].values

print(f"Shape X_train: {X_train.shape} | y_train: {y_train.shape}")

# -------------------------------------------------------------
# 7. Treinar modelo supervisionado
# -------------------------------------------------------------
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5, random_state=CONFIG['RANDOM_SEED']))
])
model.fit(X_train, y_train)
print("Modelo treinado (LogisticRegression).")

# -------------------------------------------------------------
# 8. Preparar pares de teste (positivos hidden + candidatos adicionais por usuário)
# -------------------------------------------------------------
# Para cada usuário em hidden, garantimos scoring dos produtos hidden e mais alguns extras.
hidden_users = hidden_df['user_id'].unique()

# Pré-carregar features de usuário e produto (pode usar user_feat_df / prod_feat_df; se usuário/prod não estiver no treino, tenta buscar em all_links_df)
# Map para acesso rápido
user_feat_map = user_feat_df.set_index('user_id').to_dict(orient='index')
prod_feat_map = prod_feat_df.set_index('product_id').to_dict(orient='index')

# fallback para usuários sem registro em user_feat_df
if len(user_feat_map) < len(hidden_users):
    extra_users = set(hidden_users) - set(user_feat_map.keys())
    if extra_users:
        extra_source = all_links_df[all_links_df['user_id'].isin(extra_users)]
        extra_user_feat = extra_source.groupby('user_id')[user_cols].first().reset_index()
        for _, row in extra_user_feat.iterrows():
            user_feat_map[row['user_id']] = row[user_cols].to_dict()

# fallback para produtos escondidos que não apareceram em treino
hidden_products_global = set(hidden_df['parent_asin'])
missing_products = hidden_products_global - set(prod_feat_map.keys())
if missing_products:
    extra_prod_source = all_links_df[all_links_df['parent_asin'].isin(missing_products)]
    extra_prod_feat = extra_prod_source.groupby('parent_asin')[prod_cols].first().reset_index().rename(columns={'parent_asin': 'product_id'})
    for _, row in extra_prod_feat.iterrows():
        prod_feat_map[row['product_id']] = row[prod_cols].to_dict()

# Construir pares de teste
test_rows = []
for user in hidden_users:
    # Produtos hidden para este usuário
    hidden_products_user = hidden_df[hidden_df['user_id'] == user]['parent_asin'].tolist()
    pos_train_set = user_train_pos_products.get(user, set())
    # Candidatos extras: produtos disponíveis não positivos de treino
    extra_candidates = list(available_products - pos_train_set)
    random.shuffle(extra_candidates)
    extra_candidates = extra_candidates[:CONFIG['TEST_EXTRA_CANDIDATES_PER_USER']]

    # Garantir presença dos hidden na lista final
    candidate_list = list(set(hidden_products_user) | set(extra_candidates))

    for prod in candidate_list:
        u_feat = user_feat_map.get(user)
        p_feat = prod_feat_map.get(prod)
        if u_feat is None or p_feat is None:
            # Se faltar, pula (sem features suficientes)
            continue
        row = {
            'user_id': user,
            'parent_asin': prod,
            'user_deg': u_feat['user_deg'],
            'product_deg': p_feat['product_deg'],
            'user_pagerank': u_feat['user_pagerank'],
            'product_pagerank': p_feat['product_pagerank'],
            'user_closeness': u_feat['user_closeness'],
            'product_closeness': p_feat['product_closeness'],
            'user_eig': u_feat['user_eig'],
            'product_eig': p_feat['product_eig'],
            'user_community': u_feat['user_community'],
            'product_community': p_feat['product_community'],
        }
        test_rows.append(row)

test_df = pd.DataFrame(test_rows)
if test_df.empty:
    raise ValueError("Teste vazio: verificar se hidden_links e features possuem interseção.")

# same_community para teste
test_df['same_community'] = (test_df['user_community'] == test_df['product_community']).astype(int)
test_df['deg_pa'] = test_df['user_deg'] * test_df['product_deg']
X_test = test_df[FEATURE_COLS].values

# -------------------------------------------------------------
# 9. Predizer probabilidades
# -------------------------------------------------------------
probs = model.predict_proba(X_test)[:, 1]  # probabilidade de label=1
pred_df = test_df[['user_id', 'parent_asin']].copy()
pred_df['score'] = probs
pred_df = pred_df.rename(columns={'parent_asin': 'product_id_candidato'})

# -------------------------------------------------------------
# 10. Gerar top-k por usuário
# -------------------------------------------------------------
pred_df_sorted = pred_df.sort_values(['user_id', 'score'], ascending=[True, False])
recs_topk_df = pred_df_sorted.groupby('user_id').head(CONFIG['TOP_K']).reset_index(drop=True)

# -------------------------------------------------------------
# 11. Salvar resultados
# -------------------------------------------------------------
os.makedirs(os.path.dirname(CONFIG['OUTPUT_SUPERVISED_PATH']), exist_ok=True)
recs_topk_df.to_csv(CONFIG['OUTPUT_SUPERVISED_PATH'], index=False)
print(f"Recomendações supervisionadas (top-{CONFIG['TOP_K']}) salvas em: {CONFIG['OUTPUT_SUPERVISED_PATH']}")

if CONFIG['USE_STANDARD_FILE']:
    # Também salvar no nome padrão para que evaluate_recommendations.py leia sem modificação
    os.makedirs(os.path.dirname(CONFIG['OUTPUT_STANDARD_PATH']), exist_ok=True)
    recs_topk_df.to_csv(CONFIG['OUTPUT_STANDARD_PATH'], index=False)
    print(f"Arquivo padrão para avaliação salvo em: {CONFIG['OUTPUT_STANDARD_PATH']}")

print("Resumo parâmetros supervisionado:")
print(f"  TOP_K={CONFIG['TOP_K']} | FRACTION_TRAIN={CONFIG['FRACTION_TRAIN']} | NEG_TRAIN_PER_POS={CONFIG['NEG_TRAIN_PER_POS']}")
print(f"  TEST_EXTRA_CANDIDATES_PER_USER={CONFIG['TEST_EXTRA_CANDIDATES_PER_USER']}")
print("Concluído método supervisionado.")

