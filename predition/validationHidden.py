import pandas as pd

hidden = pd.read_csv("../dataset/hidden_links.csv")
prod_proj = pd.read_csv("../dataset/products_projection.csv")

# Conjunto de produtos em hidden_links
hidden_products = set(hidden["parent_asin"].unique())

# Conjunto de produtos presentes na projeção (colunas product_u e product_v)
proj_products = set(prod_proj["product_u"]).union(set(prod_proj["product_v"]))

# Interseção e diferença
in_both = hidden_products & proj_products
only_hidden = hidden_products - proj_products

print("Produtos em hidden_links:", len(hidden_products))
print("Produtos na projeção:", len(proj_products))
print("Produtos de hidden_links que aparecem na projeção:", len(in_both))
print("Produtos de hidden_links que NUNCA aparecem na projeção:", len(only_hidden))
