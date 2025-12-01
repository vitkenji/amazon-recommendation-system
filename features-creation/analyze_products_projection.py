import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

PRODUCTS_PROJECTION_PATH = "../dataset/products_projection.csv"
FIG_DIR = "../results"
os.makedirs(FIG_DIR, exist_ok=True)

# 1) Carregar projeção de produtos
prod_proj = pd.read_csv(PRODUCTS_PROJECTION_PATH)

# 2) Histograma dos scores de Adamic-Adar e/ou Jaccard
#    (ajusta a lógica se os nomes das colunas forem diferentes)
score_cols = [c for c in prod_proj.columns
              if c.lower().startswith("aa") or c.lower().startswith("jacc")]

print("Colunas de score detectadas:", score_cols)

for col in score_cols:
    plt.figure(figsize=(6, 4))
    plt.hist(prod_proj[col], bins=50, edgecolor="black")
    plt.xlabel(col)
    plt.ylabel("Frequência")
    plt.title(f"Distribuição dos scores de {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"hist_{col}.png"))
    plt.close()

# 3) Construir grafo da projeção de produtos para medir graus
#    Assume colunas 'product_u' e 'product_v'
Gp = nx.Graph()
for _, row in prod_proj.iterrows():
    Gp.add_edge(row["product_u"], row["product_v"])

# Grau de cada produto
degrees = dict(Gp.degree())
values = list(degrees.values())

print("Número de produtos na projeção:", len(values))
print("Grau médio:", sum(values)/len(values) if values else 0)
print(
    "Percentis de grau (25, 50, 75):",
    pd.Series(values).quantile([0.25, 0.5, 0.75]).to_dict()
)

# Histograma de graus
plt.figure(figsize=(6, 4))
plt.hist(values, bins=range(0, max(values)+2), edgecolor="black", align="left")
plt.xlabel("Grau do produto na projeção")
plt.ylabel("Número de produtos")
plt.title("Distribuição de graus na projeção de produtos")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hist_degree_products_projection.png"))
plt.close()
