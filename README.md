# Amazon Recommendation System

A product recommendation system based on **Complex Network Analysis** that predicts relationships between products and users on the Amazon platform, using supervised and unsupervised machine learning techniques.

## Overview

This project was developed for the **Network Science** course and implements an intelligent system for predicting hidden links in bipartite graphs (user-product) extracted from Amazon reviews. The system combines network topology analysis with machine learning to generate highly accurate recommendations.

## Main Objective

Predict hidden links (user-product pairs not yet connected) in an Amazon review network, implementing and comparing multiple prediction strategies based on network topology and characteristics.

---

## Dataset

### ⚠️ IMPORTANT: Dataset Download Required

This project uses the **Amazon Reviews 2023 Dataset**. To reproduce this project, you must download the dataset from:

🔗 **https://amazon-reviews-2023.github.io/**

#### Dataset Download Instructions:

1. Visit: https://amazon-reviews-2023.github.io/
2. Download the product reviews and metadata files you need
3. Place the raw `.jsonl` files in the `dataset/` directory
4. The preprocessing script will automatically format and filter the data

> **Note**: The full dataset is large (~100GB+). For testing/development purposes, consider using a subset or specific product category.

#### Dataset Source
- **Official Repository**: Amazon Reviews 2023 Dataset  
- **Format**: JSONL (JSON Lines)
- **Update Frequency**: Regularly updated reviews from Amazon
- **License**: Follow the dataset's terms of use

### Data Used
- **Reviews**: User product review data
  - 100,000+ filtered reviews
  - Sentiment analysis (positive, neutral, negative, compound)
  - Users with ≥60 reviews
  - Products with ≥1200 reviews

- **Products**: Amazon product metadata
  - Main category
  - Additional categories
  - Average rating
  - Number of reviews

### Processing
```python
# Applied filters
- Remove users with < 60 reviews
- Remove products with < 1200 reviews
- Remove duplicates and null values
- Sentiment analysis (VADER SentimentIntensityAnalyzer)
```


## Project Structure

```
amazon-recommendation-system/
│
├── pre-processing/                      # Cleaning and preparation of raw data
│   └── pre-processing.py                # Filtering, normalization, sentiment analysis
│
├── network/                             # Networks in GML format (Graph Modeling Language)
│   ├── products.gml                     # Co-product network
│   ├── users.gml                        # Co-user network
│   ├── reviews_network.gml              # Complete bipartite network (users-products)
│   ├── reviews_network_train*.gml       # Training networks (different methods)
│   └── products_projection.gml          # Unipartite product projection
│
├── features-creation/                   # Extraction of topological features
│   ├── main_network_features-creation.py   # Main features (centralities)
│   ├── products-proj-features-creation.py  # Product projection features
│   ├── user_proj_features_creation.ipynb   # User projection features
│   ├── product_proj_features_creation.ipynb
│   └── analyze_products_projection.py   # Projection analysis
│
├── prediction/                          # Prediction and evaluation pipeline
│   ├── run_prediction_pipeline.py       # Main orchestrator (CONFIG)
│   ├── build_training_graph.py          # Training graph construction
│   ├── recommendations_generation.py    # Recommendations generation
│   ├── supervised_predictions.py        # Supervised model (Logistic Regression)
│   ├── evaluate_recommendations.py      # Metrics calculation
│   └── validationHidden.py              # Validation with hidden links
│
├── dataset/                             # Processed data and results
│   ├── subsample.csv                    # Data sample (user-product-review)
│   ├── new_features.csv                 # Created features (centralities, pagerank, etc)
│   ├── products_projection.csv          # Product projection data
│   ├── hidden_links*.csv                # Hidden links for testing (ground truth)
│   ├── recommendations_all*.csv         # All generated recommendations
│   ├── recommendations_topk*.csv        # Top-K recommendations
│   └── recommendations_topk_supervised.csv
│
├── results/                             # Final evaluation metrics
│   ├── evaluation_metrics.csv           # General metrics
│   ├── evaluation_metrics__degreeMinimal5.csv
│   ├── evaluation_metrics__testJaccard.csv
│   └── evaluation_metrics__testSupervised.csv
│
├── plots/                               # Visualizations and exploratory analyses
│   ├── main_network_plots.ipynb         # General network analysis
│   ├── products_proj_plots.ipynb        # Product projection visualizations
│   ├── users_proj_plots.ipynb           # User projection visualizations
│   └── visualizacao_graficoexe1.gml.gephi  # Gephi project
│
├── Article and Apresentation/           # Documentation and presentation
│   └── Relatório___Projeto_Ciência_das_Redes.pdf
│
└── README.md
```

---

## Technical Features

### Extracted Features (Topological)

| Feature | Description | Type |
|---------|-------------|------|
| `degree_centrality` | Normalized node degree | Normalized (0-1) |
| `pagerank` | PageRank importance | Probability |
| `closeness_centrality` | Average proximity | Normalized (0-1) |
| `eigenvector_centrality` | Eigenvector centrality | Normalized (0-1) |
| `clustering_coefficient` | Clustering coefficient | 0-1 |
| `common_neighbors` | Common neighbors | Count |
| `jaccard_similarity` | Jaccard index | 0-1 |
| `adamic_adar_score` | Adamic-Adar score | Numeric |
| `sentiment_score` | Sentiment score (VADER) | -1 to 1 |

### Sentiment Metrics (from preprocessing)
- `neg` - Negative proportion
- `neu` - Neutral proportion  
- `pos` - Positive proportion
- `compound` - Compound score (-1 to 1)

---

## Implemented Prediction Methods

### 1. **Unsupervised Method: Similarity**

#### Jaccard Similarity Strategy
```python
Jaccard(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
```
- Recommends products with higher Jaccard index with user
- Works well for simple similarity patterns
- Computationally fast

#### Adamic-Adar Strategy
```python
AA(u, v) = Σ 1/log(deg(w))  for w in N(u) ∩ N(v)
```
- Favors common neighbors with low degree
- Improvement over pure Jaccard
- Captures finer local structures

#### Minimum Degree Filtering
- Removes recommendations of nodes with degree < threshold
- Improves quality by excluding peripheral nodes
- Adjustable parameter (e.g., 5)

### 2. **Supervised Method: Logistic Regression**

#### Approach
- **Model**: Logistic Regression (scikit-learn)
- **Features**: All centralities + sentiment + structural
- **Balancing**: `NEGATIVE_MULTIPLIER` negatives per positive
- **Validation**: Train/test split with `TEST_FRACTION`

#### Process
```
1. Build training graph (randomly remove edges)
2. Extract features for connected (u,v) pairs (positives)
3. Generate negatives: unconnected pairs
4. Balance dataset
5. Train model
6. Predict probabilities for all unconnected pairs
7. Rank by probability
```

#### Advantages
- Learns complex patterns
- Combines multiple features
- Better performance on dense networks

---

## How to Use

### Prerequisites

Before starting, ensure you have:
1. **Python 3.8+** installed
2. **Downloaded the Amazon Reviews 2023 Dataset** from https://amazon-reviews-2023.github.io/
3. **Raw dataset files** placed in the `dataset/` directory

### Install Dependencies

```bash
pip install networkx pandas scikit-learn numpy matplotlib seaborn nltk jupyter
```

### Download NLTK Resources (first time)
```python
import nltk
nltk.download('vader_lexicon')
```

### Pipeline Configuration

Edit `prediction/run_prediction_pipeline.py` - `CONFIG` section:

```python
CONFIG = {
    # Execution control
    'SKIP_SPLIT': True,              # Skip train split
    'SKIP_PRED': False,              # Skip prediction
    'SKIP_EVAL': False,              # Skip evaluation
    
    # Method: 'similarity' or 'supervised'
    'PREDICTION_METHOD': 'supervised',
    
    # General parameters
    'TEST_FRACTION': 0.10,           # 10% of edges for testing
    'MIN_DEGREE_FOR_REMOVAL': 5,     # Remove nodes with degree < 5
    'K_LIST': [1, 3, 5, 10],         # For @k metrics
    'TOP_K': 10,                     # Top-K recommendations per user
    'OUTPUT_TAG': 'testSupervised',  # Tag to identify execution
    
    # Similarity method config
    'METRIC_COLUMN': 'jaccard',      # 'jaccard' or 'AA_score'
    'AGGREGATION': 'sum',            # 'sum' or 'mean'
    
    # Supervised method config
    'SUP_NEG_TRAIN_PER_POS': 2,      # Negatives per positive in training
    'SUP_FRACTION_TRAIN': 1.0,       # Fraction of data to train
}
```

### Run Complete Pipeline

```bash
cd prediction/
python run_prediction_pipeline.py
```

This executes sequentially:
1. ✅ Build training graph
2. ✅ Feature extraction
3. ✅ Recommendations generation
4. ✅ Evaluation

### Run Individual Components

**1. Data Preprocessing**
```bash
cd pre-processing/
python pre-processing.py
# Generates: subsample.csv with sentiment features
```

**2. Network Feature Creation**
```bash
cd features-creation/
python main_network_features-creation.py
# Generates: new_features.csv with centralities
```

**3. Exploratory Analysis (Notebooks)**
```bash
jupyter notebook plots/main_network_plots.ipynb
jupyter notebook plots/products_proj_plots.ipynb
```

---

## Results and Metrics

### Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Precision@K** | Proportion of hits in top-K | 0-1 |
| **Recall@K** | Proportion of true positives found | 0-1 |
| **F1@K** | Harmonic mean of Precision and Recall | 0-1 |
| **NDCG@K** | Normalized Discounted Cumulative Gain | 0-1 |
| **MRR** | Mean Reciprocal Rank (avg position of 1st hit) | 0-1 |
| **MAP** | Mean Average Precision | 0-1 |

### Method Comparison

Results are located in `results/`:

| File | Method | Description |
|------|--------|-------------|
| `evaluation_metrics.csv` | All | General baseline |
| `evaluation_metrics__testJaccard.csv` | Jaccard | Unsupervised similarity |
| `evaluation_metrics__degreeMinimal5.csv` | Degree Filtering | Minimum degree filtering |
| `evaluation_metrics__testSupervised.csv` | Logistic Regression | Supervised model |

### Visualize Results

```bash
# Analyze metrics
python prediction/evaluate_recommendations.py

# Compare generated recommendations
python prediction/validationHidden.py
```

---

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                 DATASET RAW (Reviews + Products)                │
│              (Download from amazon-reviews-2023.github.io/)     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    ┌────▼────────────────────┐
                    │  Pre-processing.py      │
                    │  • Filter users/products│
                    │  • Sentiment analysis   │
                    │  • Remove duplicates    │
                    └────┬──────────────────┬─┘
                         │                  │
         ┌───────────────▼─────┐  ┌────────▼──────────────┐
         │   Network Building  │  │  Features Creation   │
         │  • Bipartite graph  │  │  • Centralities      │
         │  • Projections      │  │  • Sentiment         │
         │  • GML export       │  │  • Topological       │
         └───────────────┬─────┘  └────────┬──────────────┘
                         │                  │
              ┌──────────▼─────────────────▼──────────┐
              │   Split: Train/Test Graphs            │
              │  • Remove TEST_FRACTION of edges      │
              │  • Create balanced dataset (neg/pos)  │
              └──────────────┬──────────────┬──────────┘
                             │              │
          ┌──────────────────▼──┐  ┌───────▼──────────────┐
          │  Similarity Method  │  │ Supervised Method    │
          │  • Jaccard/AA       │  │ • Logistic Reg.      │
          │  • Rank by score    │  │ • Train model        │
          │  • Get top-K        │  │ • Predict prob.      │
          └──────────────────┬──┘  └───────┬──────────────┘
                             │              │
              ┌──────────────▼──────────────▼──────────┐
              │   Combine Recommendations              │
              │  • Use TEST_FRACTION for validation    │
              └────────────────┬───────────────────────┘
                               │
              ┌────────────────▼─────────────────┐
              │   Evaluation Metrics             │
              │  • Precision/Recall@K            │
              │  • NDCG, MRR, MAP                │
              │  • Save to results/              │
              └────────────────┬─────────────────┘
                               │
                    ┌──────────▼────────────┐
                    │   RESULTS TABLE       │
                    │  evaluation_metrics   │
                    └───────────────────────┘
```

## Collaborators

- **Adryan Castro Feres** - Implementation, analysis and documentation
- **Vitor Kenji Zoppo Yamada** - Conceptualization, results analysis

## Academic Details

- **Course**: Network Science
- **Semester**: 8th Semester
- **Institution**: UTFPR (Federal Technological University of Paraná)

