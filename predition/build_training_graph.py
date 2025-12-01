import os
import random
import pandas as pd
import networkx as nx


# Parâmetros via ambiente (fallback para defaults)
test_fraction = float(os.environ.get('TEST_FRACTION', '0.10'))
min_degree_for_removal = int(os.environ.get('MIN_DEGREE_FOR_REMOVAL', '2'))
output_tag = os.environ.get('OUTPUT_TAG')
random.seed(42)

# Carrega grafo bipartido original
G = nx.read_gml('../network/reviews_network.gml')

# Calcula graus e determina arestas elegíveis para remoção segundo critério de grau
deg = dict(G.degree())
eligible_edges = [
    (u, v) for u, v in G.edges()
    if deg[u] >= min_degree_for_removal and deg[v] >= min_degree_for_removal
]

total_edges = G.number_of_edges()
target_num = int(total_edges * test_fraction)

if len(eligible_edges) < target_num:
    # Ajuste automático caso não haja arestas suficientes elegíveis
    print(f"Atenção: apenas {len(eligible_edges)} arestas elegíveis (< alvo {target_num}). Reduzindo alvo.")
    target_num = len(eligible_edges)

# Sorteia arestas de teste dentre as elegíveis
random.shuffle(eligible_edges)
test_edges = eligible_edges[:target_num]

# Função para garantir ordem user/product
def classify_edge(u, v):
    u_type = G.nodes[u].get('type')
    v_type = G.nodes[v].get('type')
    if u_type == 'user' and v_type == 'product':
        return u, v
    if u_type == 'product' and v_type == 'user':
        return v, u
    return u, v  # fallback (não esperado)

# Monta DataFrame de links escondidos
test_rows = []
for u, v in test_edges:
    user, product = classify_edge(u, v)
    test_rows.append({'user_id': user, 'parent_asin': product})
hidden_df = pd.DataFrame(test_rows)

# Cria grafo de treino sem as arestas de teste
G_train = G.copy()
G_train.remove_edges_from(test_edges)

# Salva resultados
hidden_path = '../dataset/hidden_links.csv'
train_path = '../network/reviews_network_train.gml'
if output_tag:
    base_h, ext_h = os.path.splitext(os.path.basename(hidden_path))
    base_t, ext_t = os.path.splitext(os.path.basename(train_path))
    hidden_path = os.path.join(os.path.dirname(hidden_path), f"{base_h}__{output_tag}{ext_h}")
    train_path = os.path.join(os.path.dirname(train_path), f"{base_t}__{output_tag}{ext_t}")

hidden_df.to_csv(hidden_path, index=False)
nx.write_gml(G_train, train_path)

largest_comp_size_before = len(max(nx.connected_components(G), key=len))
largest_comp_size_after = len(max(nx.connected_components(G_train), key=len))

print(f'Total de arestas (original): {total_edges}')
print(f'Arestas elegíveis (grau >= {min_degree_for_removal}): {len(eligible_edges)}')
print(f'Arestas removidas: {len(test_edges)} (fração efetiva: {len(test_edges)/total_edges:.4f})')
print(f'Maior componente antes: {largest_comp_size_before} nós | depois: {largest_comp_size_after} nós')
print(f'Links escondidos salvos em: {hidden_path}')
print(f'Grafo de treino salvo em: {train_path}')

# Explicação adicional da conectividade em termos percentuais
# Verifica a proporção de nós na maior componente conectada antes e depois, para sempre garantir que a estrutura do grafo é mantida 
perc_before = largest_comp_size_before / G.number_of_nodes()
perc_after = largest_comp_size_after / G_train.number_of_nodes()
ratio = perc_after / perc_before if perc_before > 0 else 0
status_msg = (
    'estrutura preservada' if ratio >= 0.95 else
    'leve redução; normalmente aceitável' if ratio >= 0.85 else
    'redução significativa; considere diminuir test_fraction ou min_degree_for_removal'
)
print(
    f"Conectividade: maior componente mantinha {perc_before:.2%} dos nós; após remoção mantém {perc_after:.2%}. "
    f"Retenção relativa: {ratio:.2%}. Avaliação: {status_msg}."
)