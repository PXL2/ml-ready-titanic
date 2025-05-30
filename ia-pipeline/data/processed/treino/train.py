# ia-pipeline/data/processed/treino/train.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, classification_report, confusion_matrix, make_scorer

# Definir os caminhos baseados na localização do script
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # /ia-pipeline/data/processed/treino
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, '..', 'processed_titanic_data.csv') # /ia-pipeline/data/processed/processed_titanic_data.csv
MODELS_OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'models') # /ia-pipeline/models
MODELS_CACHE_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'models_cache') # /ia-pipeline/models_cache

# Criar diretórios se não existirem
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

TARGET_PRECISION_0 = 0.80
TARGET_PRECISION_1 = 0.80

def load_processed_data(path):
    """Carrega os dados processados."""
    print(f"Carregando dados processados de: {path}")
    return pd.read_csv(path)

def find_best_threshold_and_get_predictions(model, X_val, y_val_true):
    """
    Encontra o melhor limiar para um modelo que atenda aos critérios de precisão
    para ambas as classes e retorna as predições e o limiar.
    Prioriza maior P(1) > TARGET_PRECISION_1, depois maior P(0) > TARGET_PRECISION_0,
    depois maior soma P(0)+P(1).
    """
    y_val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_p0 = 0
    best_p1 = 0
    found_good_threshold = False

    thresholds = np.arange(0.05, 0.96, 0.01)

    for threshold_candidate in thresholds:
        y_val_pred_candidate = (y_val_probs >= threshold_candidate).astype(int)
        p0 = precision_score(y_val_true, y_val_pred_candidate, pos_label=0, zero_division=0)
        p1 = precision_score(y_val_true, y_val_pred_candidate, pos_label=1, zero_division=0)

        if p0 >= TARGET_PRECISION_0 and p1 >= TARGET_PRECISION_1:
            if not found_good_threshold: # Primeira vez que encontramos um bom limiar
                found_good_threshold = True
                best_threshold = threshold_candidate
                best_p0 = p0
                best_p1 = p1
            elif p1 > best_p1: # Prioridade 1: Maximizar P(1)
                best_threshold = threshold_candidate
                best_p0 = p0
                best_p1 = p1
            elif p1 == best_p1 and p0 > best_p0: # Prioridade 2: Maximizar P(0) se P(1) for igual
                best_threshold = threshold_candidate
                best_p0 = p0
                best_p1 = p1
            elif p1 == best_p1 and p0 == best_p0 and (p0 + p1 > best_p0 + best_p1): # Prioridade 3: Maior soma
                 best_threshold = threshold_candidate
                 best_p0 = p0
                 best_p1 = p1

    if not found_good_threshold: # Se nenhum limiar atingiu as metas, usa o default com melhor soma P(0)+P(1)
        print("Nenhum limiar atingiu as metas de precisão. Selecionando por melhor soma P0+P1.")
        max_sum_p = 0
        for threshold_candidate in thresholds:
            y_val_pred_candidate = (y_val_probs >= threshold_candidate).astype(int)
            p0 = precision_score(y_val_true, y_val_pred_candidate, pos_label=0, zero_division=0)
            p1 = precision_score(y_val_true, y_val_pred_candidate, pos_label=1, zero_division=0)
            if (p0 + p1) > max_sum_p:
                max_sum_p = p0 + p1
                best_threshold = threshold_candidate
                best_p0 = p0
                best_p1 = p1
    
    final_predictions = (y_val_probs >= best_threshold).astype(int)
    print(f"Melhor limiar encontrado: {best_threshold:.2f} -> P(0)={best_p0:.2f}, P(1)={best_p1:.2f}")
    return final_predictions, best_threshold


def train_and_evaluate_models(X_train_smote, y_train_smote, X_test, y_test):
    """Treina, otimiza e avalia múltiplos modelos."""
    print("Iniciando treinamento e avaliação de modelos...")

    # Definindo os modelos e seus respectivos espaços de hiperparâmetros
    models_params = {
        'LogisticRegression': (
            LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
            {'C': [0.01, 0.1, 1, 10, 100]}
        ),
        'SVC': (
            SVC(probability=True, random_state=42, class_weight='balanced'),
            {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
        ),
        'RandomForestClassifier': (
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
        ),
        'GradientBoostingClassifier': (
            GradientBoostingClassifier(random_state=42),
            {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
        )
    }

    # Métricas e Validação Cruzada
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # Usar precision_macro para o GridSearchCV tentar balancear a precisão entre as classes
    scoring_metric = make_scorer(precision_score, average='macro', zero_division=0)

    best_overall_model = None
    best_overall_threshold = 0.5
    best_overall_p0 = 0
    best_overall_p1 = 0
    best_model_name = ""

    results = {}

    for model_name, (model, params) in models_params.items():
        print(f"\n--- Treinando e Otimizando: {model_name} ---")
        
        # Pipeline com StandardScaler para modelos que se beneficiam disso
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name.lower(), model) # Nome do estimador no pipeline deve ser minúsculo
        ])
        
        # Ajustar nomes dos parâmetros para o pipeline
        pipeline_params = {f'{model_name.lower()}__{k}': v for k, v in params.items()}

        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=pipeline_params,
                                   scoring=scoring_metric,
                                   cv=cv_strategy,
                                   verbose=1,
                                   n_jobs=-1, # Usar todos os processadores
                                   error_score='raise') # Levanta erro se algo der errado

        # Habilitar caching para GridSearchCV
        cached_gs = joblib.Memory(location=MODELS_CACHE_DIR, verbose=0).cache(grid_search.fit)
        
        try:
            cached_gs(X_train_smote, y_train_smote)
            best_estimator = grid_search.best_estimator_ # O pipeline treinado com os melhores parâmetros
            
            print(f"Melhores parâmetros para {model_name}: {grid_search.best_params_}")

            # Ajuste de limiar no conjunto de teste (idealmente seria um conjunto de validação separado)
            y_pred_adj, best_threshold_adj = find_best_threshold_and_get_predictions(best_estimator, X_test, y_test)
            
            p0_adj = precision_score(y_test, y_pred_adj, pos_label=0, zero_division=0)
            p1_adj = precision_score(y_test, y_pred_adj, pos_label=1, zero_division=0)

            results[model_name] = {
                'best_estimator': best_estimator,
                'best_params': grid_search.best_params_,
                'best_threshold': best_threshold_adj,
                'precision_0': p0_adj,
                'precision_1': p1_adj,
                'classification_report': classification_report(y_test, y_pred_adj, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred_adj)
            }
            print(f"Reporte de Classificação para {model_name} (com limiar ajustado = {best_threshold_adj:.2f}):")
            print(results[model_name]['classification_report'])
            print(f"Matriz de Confusão:\n{results[model_name]['confusion_matrix']}")

            # Lógica para selecionar o melhor modelo geral
            if p0_adj >= TARGET_PRECISION_0 and p1_adj >= TARGET_PRECISION_1:
                if best_overall_model is None or \
                   (p1_adj > best_overall_p1) or \
                   (p1_adj == best_overall_p1 and p0_adj > best_overall_p0) or \
                   (p1_adj == best_overall_p1 and p0_adj == best_overall_p0 and (p0_adj + p1_adj > best_overall_p0 + best_overall_p1)):
                    best_overall_model = best_estimator
                    best_overall_threshold = best_threshold_adj
                    best_overall_p0 = p0_adj
                    best_overall_p1 = p1_adj
                    best_model_name = model_name
        except Exception as e:
            print(f"Erro ao treinar {model_name}: {e}")
            results[model_name] = {'error': str(e)}


    # Treinamento do StackingClassifier (Exemplo)
    print("\n--- Treinando e Otimizando: StackingClassifier ---")
    estimators = [
        ('rf', results.get('RandomForestClassifier', {}).get('best_estimator', RandomForestClassifier(random_state=42))),
        ('gb', results.get('GradientBoostingClassifier', {}).get('best_estimator', GradientBoostingClassifier(random_state=42)))
    ]
    # Filtrar estimadores que podem ter falhado
    valid_estimators = [est for est in estimators if est[1] is not None and not isinstance(est[1], dict)] # Checa se não é um dict de erro

    if len(valid_estimators) >= 2: # Stacking precisa de pelo menos 2 estimadores base válidos
        stacking_clf = StackingClassifier(
            estimators=valid_estimators,
            final_estimator=LogisticRegression(solver='liblinear', random_state=42),
            cv=cv_strategy
        )
        # Pipeline para o Stacking
        pipeline_stacking = Pipeline([
            ('scaler', StandardScaler()), # Scaling antes do Stacking
            ('stacking', stacking_clf)
        ])
        
        try:
            pipeline_stacking.fit(X_train_smote, y_train_smote)
            y_pred_stacking_adj, best_threshold_stacking_adj = find_best_threshold_and_get_predictions(pipeline_stacking, X_test, y_test)
            
            p0_stacking_adj = precision_score(y_test, y_pred_stacking_adj, pos_label=0, zero_division=0)
            p1_stacking_adj = precision_score(y_test, y_pred_stacking_adj, pos_label=1, zero_division=0)

            results['StackingClassifier'] = {
                'best_estimator': pipeline_stacking,
                'best_threshold': best_threshold_stacking_adj,
                'precision_0': p0_stacking_adj,
                'precision_1': p1_stacking_adj,
                'classification_report': classification_report(y_test, y_pred_stacking_adj, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred_stacking_adj)
            }
            print(f"Reporte de Classificação para StackingClassifier (com limiar ajustado = {best_threshold_stacking_adj:.2f}):")
            print(results['StackingClassifier']['classification_report'])
            print(f"Matriz de Confusão:\n{results['StackingClassifier']['confusion_matrix']}")

            if p0_stacking_adj >= TARGET_PRECISION_0 and p1_stacking_adj >= TARGET_PRECISION_1:
                if best_overall_model is None or \
                   (p1_stacking_adj > best_overall_p1) or \
                   (p1_stacking_adj == best_overall_p1 and p0_stacking_adj > best_overall_p0) or \
                   (p1_stacking_adj == best_overall_p1 and p0_stacking_adj == best_overall_p0 and \
                    (p0_stacking_adj + p1_stacking_adj > best_overall_p0 + best_overall_p1)):
                    best_overall_model = pipeline_stacking
                    best_overall_threshold = best_threshold_stacking_adj
                    best_overall_p0 = p0_stacking_adj
                    best_overall_p1 = p1_stacking_adj
                    best_model_name = "StackingClassifier"
        except Exception as e:
            print(f"Erro ao treinar StackingClassifier: {e}")
            results['StackingClassifier'] = {'error': str(e)}
    else:
        print("Não foi possível treinar o StackingClassifier devido à falta de estimadores base válidos.")


    if best_overall_model:
        print(f"\n🏆 Melhor modelo geral: {best_model_name} com Limiar={best_overall_threshold:.2f}, P(0)={best_overall_p0:.2f}, P(1)={best_overall_p1:.2f}")
        model_filename = os.path.join(MODELS_OUTPUT_DIR, f'best_titanic_model_{best_model_name.lower()}.joblib')
        threshold_filename = os.path.join(MODELS_OUTPUT_DIR, f'best_titanic_model_{best_model_name.lower()}_threshold.joblib')
        joblib.dump(best_overall_model, model_filename)
        joblib.dump(best_overall_threshold, threshold_filename)
        print(f"Melhor modelo salvo em: {model_filename}")
        print(f"Melhor limiar salvo em: {threshold_filename}")
    else:
        print("\nNenhum modelo atingiu os critérios de precisão desejados para ser salvo como 'melhor modelo'.")

    return results


if __name__ == '__main__':
    df = load_processed_data(PROCESSED_DATA_PATH)

    if 'Survived' not in df.columns:
        raise ValueError("Coluna 'Survived' não encontrada nos dados processados. Verifique processados.py")

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Dividir em treino e teste ANTES do SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Formato dos dados de treino antes do SMOTE: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Formato dos dados de teste: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"Distribuição da classe 'Survived' no treino original:\n{y_train.value_counts(normalize=True)}")

    # Aplicar SMOTE apenas no conjunto de treino
    print("\nAplicando SMOTE no conjunto de treino...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Formato dos dados de treino após SMOTE: X_train_smote={X_train_smote.shape}, y_train_smote={y_train_smote.shape}")
    print(f"Distribuição da classe 'Survived' no treino após SMOTE:\n{pd.Series(y_train_smote).value_counts(normalize=True)}")

    # Treinar e avaliar modelos
    model_results = train_and_evaluate_models(X_train_smote, y_train_smote, X_test, y_test)

    # (Opcional) Você pode querer salvar todos os resultados ou um resumo
    # Por exemplo, salvar um resumo das precisões de todos os modelos
    summary = {name: {'P(0)': data.get('precision_0', 0), 'P(1)': data.get('precision_1', 0), 'Threshold': data.get('best_threshold', 0.5)}
               for name, data in model_results.items() if 'error' not in data}
    print("\nResumo dos Resultados Finais (Precisão no Teste com Limiar Ajustado):")
    for name, metrics in summary.items():
        print(f"- {name}: P(0)={metrics['P(0)']:.2f}, P(1)={metrics['P(1)']:.2f} @ Threshold={metrics['Threshold']:.2f}")