import os
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# -------------------------------------------------------------
# CONFIGURAÇÕES AJUSTÁVEIS
# -------------------------------------------------------------
# Lista de valores de k para avaliar precision@k e recall@k.
_k_env = os.environ.get('K_LIST')
K_LIST = [int(k) for k in _k_env.split()] if _k_env else [1, 3, 5, 10, 20, 50]
OUTPUT_TAG = os.environ.get('OUTPUT_TAG')
TOPK_RECS_PATH = '../dataset/recommendations_topk.csv'
ALL_RECS_PATH = '../dataset/recommendations_all.csv'  # não vo usar (arquivo muito grande)
HIDDEN_LINKS_PATH = '../dataset/hidden_links.csv' # Gabarito de links escondidos
if OUTPUT_TAG:
    def suffixed(path):
        base, ext = os.path.splitext(os.path.basename(path))
        return os.path.join(os.path.dirname(path), f"{base}__{OUTPUT_TAG}{ext}")
    TOPK_RECS_PATH = suffixed(TOPK_RECS_PATH)
    ALL_RECS_PATH = suffixed(ALL_RECS_PATH)
    HIDDEN_LINKS_PATH = suffixed(HIDDEN_LINKS_PATH)

# Saídas
METRICS_CSV_PATH = '../results/evaluation_metrics.csv'
METRICS_JSON_PATH = '../results/evaluation_metrics.json'
FIG_DIR = '../results'
if OUTPUT_TAG:
    base_csv, ext_csv = os.path.splitext(os.path.basename(METRICS_CSV_PATH))
    base_json, ext_json = os.path.splitext(os.path.basename(METRICS_JSON_PATH))
    METRICS_CSV_PATH = os.path.join(os.path.dirname(METRICS_CSV_PATH), f"{base_csv}__{OUTPUT_TAG}{ext_csv}")
    METRICS_JSON_PATH = os.path.join(os.path.dirname(METRICS_JSON_PATH), f"{base_json}__{OUTPUT_TAG}{ext_json}")
    # Para figuras, salva com sufixo diretamente nos nomes
    FIG_DIR = '../results'
# Diretório para figuras
# Quantidade de amostras negativas por positivo (ex.: 3 => triplica quantidade de negativos)
NEGATIVE_MULTIPLIER = int(os.environ.get('NEGATIVE_MULTIPLIER', '3')) 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------------------------------------------
# 1. Carregar dados base (hidden links e recomendações)
# -------------------------------------------------------------
if not os.path.exists(TOPK_RECS_PATH):
    raise FileNotFoundError(f'Arquivo de recomendações top-k não encontrado: {TOPK_RECS_PATH}')
if not os.path.exists(HIDDEN_LINKS_PATH):
    raise FileNotFoundError(f'Arquivo de links escondidos não encontrado: {HIDDEN_LINKS_PATH}')

hidden_df = pd.read_csv(HIDDEN_LINKS_PATH)  # colunas esperadas: user_id, parent_asin
recs_topk_df = pd.read_csv(TOPK_RECS_PATH)  # colunas: user_id, product_id_candidato, score

# Opcional: usar recomendações completas para melhor cobertura de scores
# if os.path.exists(ALL_RECS_PATH):
#     recs_all_df = pd.read_csv(ALL_RECS_PATH)  # colunas: user_id, product_id_candidato, score
# else:
#     recs_all_df = recs_topk_df.copy()
recs_all_df = recs_topk_df.copy()

# Normaliza nomes das colunas para consistência interna
hidden_df = hidden_df.rename(columns={'parent_asin': 'product_id'})
recs_topk_df = recs_topk_df.rename(columns={'product_id_candidato': 'product_id'})
recs_all_df = recs_all_df.rename(columns={'product_id_candidato': 'product_id'})

# -------------------------------------------------------------
# 2. Organizar dados por usuário
# -------------------------------------------------------------
# Gabarito: produtos escondidos por usuário
hidden_by_user = hidden_df.groupby('user_id')['product_id'].apply(set).to_dict()
# Recomendações top-k (já ordenadas pela geração); se não ordenadas, ordena por score desc
recs_topk_df = recs_topk_df.sort_values(['user_id', 'score'], ascending=[True, False])
# Evita DeprecationWarning: usa group_keys=False para não operar nas colunas de agrupamento
recs_by_user_topk = recs_topk_df.groupby('user_id', group_keys=False)['product_id'].apply(list).to_dict()
# Mapa de scores completos (user, product) -> score
score_lookup = {(row.user_id, row.product_id): row.score for row in recs_all_df.itertuples()}

# Usuários envolvidos (interseção para avaliação)
users_all = sorted(set(hidden_by_user.keys()) & set(recs_by_user_topk.keys()))
if not users_all:
    raise ValueError('Nenhum usuário em comum entre hidden_links e recomendações top-k.')

# -------------------------------------------------------------
# 3. Funções de cálculo de métricas por k
# -------------------------------------------------------------

def precision_recall_at_k(k):
    prec_list = []
    rec_list = []
    for u in users_all:
        hidden_set = hidden_by_user.get(u, set())
        if not hidden_set:
            # Se usuário não tem links escondidos, pula ou conta como 0? Aqui pulamos para não diluir média.
            continue
        recs_u = recs_by_user_topk.get(u, [])[:k]
        hits = sum(1 for p in recs_u if p in hidden_set)
        precision = hits / k if k > 0 else 0.0
        recall = hits / len(hidden_set) if hidden_set else 0.0
        prec_list.append(precision)
        rec_list.append(recall)
    # Média entre usuários com hidden links
    avg_precision = np.mean(prec_list) if prec_list else 0.0
    avg_recall = np.mean(rec_list) if rec_list else 0.0
    return avg_precision, avg_recall

# Histograma de hits com k máximo (último valor de K_LIST, supõe ser maior ou igual ao k usado em geração)
K_MAX = max(K_LIST)
user_hits = []
for u in users_all:
    hidden_set = hidden_by_user.get(u, set())
    if not hidden_set:
        continue
    recs_u = recs_by_user_topk.get(u, [])[:K_MAX]
    hits = sum(1 for p in recs_u if p in hidden_set)
    user_hits.append(hits)

# -------------------------------------------------------------
# 4. Construir base para AUC (positivos e negativos)
# -------------------------------------------------------------
# Positivos: todos os (u,p) em hidden_df que têm score disponível (se não tiver, atribuímos score mínimo 0)
positives = []
for u, hidden_set in hidden_by_user.items():
    for p in hidden_set:
        score = score_lookup.get((u, p), 0.0)  # se não geramos score para esse par, assume 0
        positives.append((u, p, score, 1))

# Negativos: amostra de produtos não escondidos e não necessariamente recomendados.
# Estratégia: para cada positivo, sorteia NEGATIVE_MULTIPLIER produtos aleatórios diferentes.
all_products_scored = set(recs_all_df['product_id'].unique())
negatives = []
for u, hidden_set in hidden_by_user.items():
    # Produtos candidatos para negativos: todos com score para esse usuário ou todos globalmente?
    # Usaremos conjunto global de produtos com score, removendo os hidden do usuário.
    candidate_products = list(all_products_scored - hidden_set)
    if not candidate_products:
        continue
    # Quantidade de negativos a gerar para este usuário
    num_pos_u = len(hidden_set)
    num_neg_u = num_pos_u * NEGATIVE_MULTIPLIER
    sampled = random.sample(candidate_products, min(num_neg_u, len(candidate_products)))
    for p in sampled:
        score = score_lookup.get((u, p), 0.0)
        negatives.append((u, p, score, 0))

# Combina e prepara para ROC/AUC
all_pairs = positives + negatives
if not all_pairs:
    raise ValueError('Não foi possível montar pares para AUC (verifique dados).')

scores = [s for (_, _, s, _) in all_pairs]
labels = [l for (_, _, _, l) in all_pairs]
auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float('nan')

fpr, tpr, _ = roc_curve(labels, scores) if len(set(labels)) == 2 else ([], [], [])

# -------------------------------------------------------------
# 5. Calcular métricas para cada k e salvar
# -------------------------------------------------------------
metrics_records = []
for k in K_LIST:
    p_at_k, r_at_k = precision_recall_at_k(k)
    metrics_records.append({'metric': f'precision@{k}', 'value': p_at_k})
    metrics_records.append({'metric': f'recall@{k}', 'value': r_at_k})

metrics_records.append({'metric': 'AUC', 'value': auc})
metrics_df = pd.DataFrame(metrics_records)
metrics_df.to_csv(METRICS_CSV_PATH, index=False)

# with open(METRICS_JSON_PATH, 'w', encoding='utf-8') as fjson:
#     json.dump({row.metric: row.value for row in metrics_df.itertuples()}, fjson, ensure_ascii=False, indent=2)

print('Resumo das métricas:')
print(metrics_df)
print(f'Métricas salvas em: {METRICS_CSV_PATH} e {METRICS_JSON_PATH}')

# -------------------------------------------------------------
# 6. Gerar visualizações
# -------------------------------------------------------------
os.makedirs(FIG_DIR, exist_ok=True)
fig_suffix = f"__{OUTPUT_TAG}" if OUTPUT_TAG else ""

# a) Histograma de acertos por usuário (usando K_MAX)
plt.figure(figsize=(6,4))
plt.hist(user_hits, bins=range(0, K_MAX+2), edgecolor='black', align='left')
plt.xlabel(f'Número de acertos no top-{K_MAX}')
plt.ylabel('Quantidade de usuários')
plt.title('Histograma de acertos por usuário')
plt.xticks(range(0, K_MAX+1))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f'histogram_hits_per_user{fig_suffix}.png'))
plt.close()

# b) Gráfico de barras precision@k e recall@k
precision_vals = []
recall_vals = []
for k in K_LIST:
    p_at_k, r_at_k = precision_recall_at_k(k)
    precision_vals.append(p_at_k)
    recall_vals.append(r_at_k)

x = np.arange(len(K_LIST))
width = 0.35
plt.figure(figsize=(7,4))
plt.bar(x - width/2, precision_vals, width, label='Precision')
plt.bar(x + width/2, recall_vals, width, label='Recall')
plt.xticks(x, [str(k) for k in K_LIST])
plt.xlabel('k')
plt.ylabel('Valor médio')
plt.title('Precision@k e Recall@k')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f'precision_recall_vs_k{fig_suffix}.png'))
plt.close()

# c) Curva ROC
if len(fpr) > 0 and len(tpr) > 0:
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.4f})')
    plt.plot([0,1], [0,1], 'k--', label='Aleatório')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'roc_curve{fig_suffix}.png'))
    plt.close()
else:
    print('Curva ROC não gerada (labels não possuem ambas as classes).')

# -------------------------------------------------------------
# 7. Resumo final
# -------------------------------------------------------------
print('Avaliação concluída.')
print(f'Figuras salvas em: {FIG_DIR}')
print('Parâmetros usados:')
print(f'  K_LIST={K_LIST}')
print(f'  NEGATIVE_MULTIPLIER={NEGATIVE_MULTIPLIER}')
print(f'  SEED={SEED}')
