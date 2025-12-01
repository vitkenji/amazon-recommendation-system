import subprocess
import sys
import os

"""
Configuração central do pipeline.
Altere os valores aqui e execute este arquivo.
"""

CONFIG = {
    # =====================
    # Controle de execução
    # =====================
    'SKIP_SPLIT': True,               # pular etapa de split/treino
    'SKIP_PRED': False,                # pular etapa de predição 
    'SKIP_EVAL': False,                # pular etapa de avaliação

    # Seleção do método de predição
    'PREDICTION_METHOD': 'supervised', # 'similarity' ou 'supervised'

    # =====================
    # Parâmetros gerais
    # =====================
    'TEST_FRACTION': 0.10,             # fração de arestas para teste
    'MIN_DEGREE_FOR_REMOVAL': 5,       # grau mínimo dos nós para remoção
    'K_LIST': [1, 3, 5, 10],           # ks para métricas
    'NEGATIVE_MULTIPLIER': 3,          # negativos por positivo
    'OUTPUT_TAG': 'testSupervised',    # se None, cria automático; caso contrário usa o fornecido

    # =====================
    # Config: método Similaridade
    # =====================
    'METRIC_COLUMN': 'jaccard',        # 'AA_score' ou 'jaccard'
    'AGGREGATION': 'sum',              # 'sum' ou 'mean'
    'TOP_K': 10,                       # top-k por usuário (similaridade)

    # =====================
    # Config: método Supervisionado
    # =====================
    'SUP_TOP_K': 10,
    'SUP_NEG_TRAIN_PER_POS': 2,
    'SUP_TEST_EXTRA': 100,
    'SUP_FRACTION_TRAIN': 1.0,
    'SUP_USE_STANDARD_FILE': True,
}

def main():
    # Define OUTPUT_TAG (automático se não definido)
    if CONFIG['OUTPUT_TAG'] is None:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        CONFIG['OUTPUT_TAG'] = (
            f"tf{CONFIG['TEST_FRACTION']}_deg{CONFIG['MIN_DEGREE_FOR_REMOVAL']}"
            f"_{CONFIG['METRIC_COLUMN']}_{CONFIG['AGGREGATION']}_k{CONFIG['TOP_K']}"
            f"_neg{CONFIG['NEGATIVE_MULTIPLIER']}_{ts}"
        )
    # 1) Split (build_training_graph.py)
    if not CONFIG['SKIP_SPLIT']:
        print('[Pipeline] Executando split da rede (build_training_graph.py)...')
        split_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'build_training_graph.py')
        ]
        # build_training_graph.py usa caminhos fixos no script; os parâmetros principais são a fração e grau mínimo
        # Para manter reaproveitamento, podemos temporariamente setar via variáveis de ambiente
        env = os.environ.copy()
        env['TEST_FRACTION'] = str(CONFIG['TEST_FRACTION'])
        env['MIN_DEGREE_FOR_REMOVAL'] = str(CONFIG['MIN_DEGREE_FOR_REMOVAL'])
        env['OUTPUT_TAG'] = CONFIG['OUTPUT_TAG']
        subprocess.run(split_cmd, env=env, check=True)

    # 2) Recomendações por similaridade (recommendations_generation.py)
    if not CONFIG['SKIP_PRED'] and CONFIG['PREDICTION_METHOD'] == 'similarity':
        print('[Pipeline] Gerando recomendações (recommendations_generation.py)...')
        recs_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'recommendations_generation.py')
        ]
        env = os.environ.copy()
        env['METRIC_COLUMN'] = CONFIG['METRIC_COLUMN']
        env['AGGREGATION'] = CONFIG['AGGREGATION']
        env['TOP_K'] = str(CONFIG['TOP_K'])
        # Para evitar 'No space left on device' ao salvar o arquivo completo
        env['SAVE_ALL'] = '0'
        env['OUTPUT_TAG'] = CONFIG['OUTPUT_TAG']
        subprocess.run(recs_cmd, env=env, check=True)

    # 3) Método supervisionado (supervised_predictions.py)
    if not CONFIG['SKIP_PRED'] and CONFIG['PREDICTION_METHOD'] == 'supervised':
        print('[Pipeline] Executando método supervisionado (supervised_predictions.py)...')
        sup_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'supervised_predictions.py')
        ]
        env = os.environ.copy()
        env['OUTPUT_TAG'] = CONFIG['OUTPUT_TAG']
        # Passa parâmetros principais do método supervisionado via ambiente
        env['SUP_TOP_K'] = str(CONFIG['SUP_TOP_K'])
        env['SUP_NEG_TRAIN_PER_POS'] = str(CONFIG['SUP_NEG_TRAIN_PER_POS'])
        env['SUP_TEST_EXTRA'] = str(CONFIG['SUP_TEST_EXTRA'])
        env['SUP_FRACTION_TRAIN'] = str(CONFIG['SUP_FRACTION_TRAIN'])
        env['SUP_USE_STANDARD_FILE'] = '1' if CONFIG['SUP_USE_STANDARD_FILE'] else '0'
        subprocess.run(sup_cmd, env=env, check=True)

    # 4) Avaliação (evaluate_recommendations.py)
    if not CONFIG['SKIP_EVAL']:
        print('[Pipeline] Avaliando recomendações (evaluate_recommendations.py)...')
        eval_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'evaluate_recommendations.py')
        ]
        env = os.environ.copy()
        env['K_LIST'] = ' '.join(map(str, CONFIG['K_LIST']))
        env['NEGATIVE_MULTIPLIER'] = str(CONFIG['NEGATIVE_MULTIPLIER'])
        env['OUTPUT_TAG'] = CONFIG['OUTPUT_TAG']
        subprocess.run(eval_cmd, env=env, check=True)

    print('Pipeline concluído.')


if __name__ == '__main__':
    main()
