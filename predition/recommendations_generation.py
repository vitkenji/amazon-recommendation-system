import os
import pandas as pd
import networkx as nx
from collections import defaultdict


METRIC_COLUMN = os.environ.get('METRIC_COLUMN', 'AA_score')
AGGREGATION = os.environ.get('AGGREGATION', 'sum')
TOP_K = int(os.environ.get('TOP_K', '10')) 
OUTPUT_TAG = os.environ.get('OUTPUT_TAG')

PRODUCTS_PROJECTION_PATH = '../dataset/products_projection.csv'
TRAIN_GRAPH_PATH = '../network/reviews_network_train.gml'
OUTPUT_ALL_PATH = '../dataset/recommendations_all.csv'
OUTPUT_TOPK_PATH = '../dataset/recommendations_topk.csv'
SAVE_ALL = os.environ.get('SAVE_ALL', '0')  # '1' para salvar completo; '0' para pular
if OUTPUT_TAG:
    base_all, ext_all = os.path.splitext(os.path.basename(OUTPUT_ALL_PATH))
    base_topk, ext_topk = os.path.splitext(os.path.basename(OUTPUT_TOPK_PATH))
    OUTPUT_ALL_PATH = os.path.join(os.path.dirname(OUTPUT_ALL_PATH), f"{base_all}__{OUTPUT_TAG}{ext_all}")
    OUTPUT_TOPK_PATH = os.path.join(os.path.dirname(OUTPUT_TOPK_PATH), f"{base_topk}__{OUTPUT_TAG}{ext_topk}")


print('Carregando projeção de produtos e grafo de treino...')
prod_proj_df = pd.read_csv(PRODUCTS_PROJECTION_PATH)
G_train = nx.read_gml(TRAIN_GRAPH_PATH)

# Validação básica das colunas esperadas
expected_cols = {'product_u', 'product_v', METRIC_COLUMN}
if not expected_cols.issubset(prod_proj_df.columns):
    raise ValueError(f'Colunas esperadas ausentes em products_projection.csv. Esperado mínimo: {expected_cols}. Encontrado: {prod_proj_df.columns.tolist()}')

# -------------------------------------------------------------
# 2. Construir mapa de similaridades produto -> (outro_produto, score)
# -------------------------------------------------------------
# Estrutura: similarities[produto] = lista de (produto_similar, score)
# Serve para acesso rápido durante geração de recomendações
similarities = defaultdict(list)
for _, row in prod_proj_df.iterrows():
    u = row['product_u']
    v = row['product_v']
    score = row[METRIC_COLUMN]
    # Adiciona nas duas direções (grafo não orientado)
    similarities[u].append((v, score))
    similarities[v].append((u, score))

print(f'Mapa de similaridades criado para {len(similarities)} produtos usando métricas em {METRIC_COLUMN}.')

# -------------------------------------------------------------
# 3. Identificar usuários e seus produtos avaliados na rede de treino
# -------------------------------------------------------------
# Assume atributo 'type' nos nós: 'user' ou 'product'
users = [n for n, d in G_train.nodes(data=True) if d.get('type') == 'user']
products = set(n for n, d in G_train.nodes(data=True) if d.get('type') == 'product')

user_products_train = {}
for u in users:
    # Vizinhos que são produtos
    rated = [nbr for nbr in G_train.neighbors(u) if G_train.nodes[nbr].get('type') == 'product']
    user_products_train[u] = set(rated)

print(f'Usuários encontrados: {len(users)} | Produtos encontrados: {len(products)}')


# -------------------------------------------------------------
# 4. Gerar candidatos por usuário (produtos similares aos que ele já avaliou)
# -------------------------------------------------------------
# Para cada usuário: acumula scores de produtos similares não avaliados
rows = []
for user in users:
    rated = user_products_train[user]
    candidate_scores = defaultdict(float)
    counts = defaultdict(int)  # usado para media

    for p in rated:
        # Para cada produto avaliado, pega seus similares
        for sim_prod, score in similarities.get(p, []):
            # Ignora produtos já avaliados
            if sim_prod in rated:
                continue
            # Ignora se sim_prod não está no conjunto de produtos do grafo (segurança)
            if sim_prod not in products:
                continue
            candidate_scores[sim_prod] += score
            counts[sim_prod] += 1

    # Converte agregação
    for cand, agg_score in candidate_scores.items():
        if AGGREGATION == 'mean':
            final_score = agg_score / counts[cand]
        else:  # sum
            final_score = agg_score
        rows.append({'user_id': user, 'product_id_candidato': cand, 'score': final_score})

# -------------------------------------------------------------
# 5. Montar DataFrame e salvar completo
# -------------------------------------------------------------
recs_df = pd.DataFrame(rows)
if recs_df.empty:
    print('Nenhuma recomendação gerada (verifique se há similaridades ou se parâmetros estão muito restritos).')
else:
    print(f'Total de linhas de recomendações: {len(recs_df)}')

# Salva todas as recomendações (antes de filtrar top-k) — opcional
if SAVE_ALL == '1' and not recs_df.empty:
    # cria diretório se necessário
    os.makedirs(os.path.dirname(OUTPUT_ALL_PATH), exist_ok=True)
    # dica: pode comprimir para reduzir espaço
    recs_df.to_csv(OUTPUT_ALL_PATH, index=False)
    print(f'Recomendações completas salvas em: {OUTPUT_ALL_PATH}')
else:
    print('Arquivo completo de recomendações não salvo (SAVE_ALL=0).')

# -------------------------------------------------------------
# 6. Filtrar top-k por usuário
# -------------------------------------------------------------
if not recs_df.empty:
    recs_df_sorted = recs_df.sort_values(['user_id', 'score'], ascending=[True, False])
    topk_df = recs_df_sorted.groupby('user_id').head(TOP_K).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUTPUT_TOPK_PATH), exist_ok=True)
    topk_df.to_csv(OUTPUT_TOPK_PATH, index=False)
    print(f'Top-{TOP_K} recomendações por usuário salvas em: {OUTPUT_TOPK_PATH}')

# -------------------------------------------------------------
# 7. Resumo final
# -------------------------------------------------------------
print('Resumo parâmetros:')
print(f'  METRIC_COLUMN={METRIC_COLUMN} | AGGREGATION={AGGREGATION} | TOP_K={TOP_K}')
print('Concluído.')
