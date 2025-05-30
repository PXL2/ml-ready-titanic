from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import re # Para a função extract_title

app = Flask(__name__)

# --- Configuração e Constantes ---

# Caminho para a pasta onde o modelo e o limiar estão salvos
# Assumindo que app.py está em ia-pipeline/src/api/app.py
# e os modelos estão em ia-pipeline/models/
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')

# Nomes dos arquivos baseados na saída do seu script train.py
BEST_MODEL_FILENAME = 'best_titanic_model_plots_Logistic_Regression.joblib'
BEST_THRESHOLD_FILENAME = 'best_titanic_model_plots_Logistic_Regression_threshold.joblib' # CONFIRME ESTE NOME DE ARQUIVO

MODEL_PATH = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
THRESHOLD_PATH = os.path.join(MODEL_DIR, BEST_THRESHOLD_FILENAME)

# Valores de fallback para imputação (idealmente, viriam do treino)
AGE_MEDIAN = 29.0
FARE_MEDIAN = 14.45
EMBARKED_MODE = 'S' # O valor mais comum

# Mapeamentos para Label Encoding (DEVEM CORRESPONDER AO TREINO)
SEX_MAP = {'male': 0, 'female': 1}
EMBARKED_MAP = {'S': 0, 'C': 1, 'Q': 2} # Verifique se esta ordem está correta!
# Títulos simplificados conforme processados.py (verifique se o mapeamento numérico está correto)
TITLE_MAP = {
    'Mr': 0,
    'Miss': 1,
    'Mrs': 2,
    'Master': 3,
    'Rare': 4 # Agrupamento de títulos raros
}

# Ordem final das features que o pipeline do modelo espera (APÓS pré-processamento inicial, ANTES do StandardScaler)
MODEL_FEATURES_ORDER = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked', 'Title', 'FamilySize', 'IsAlone', 'HasCabin', 'FarePerPerson'
]

# --- Carregar Modelo e Limiar ---
# ADICIONAR LINHAS DE DEBUG AQUI:
print(f"--- INÍCIO DEBUG CAMINHOS ---")
print(f"Diretório de execução do script (app.py): {os.path.dirname(os.path.abspath(__file__))}")
print(f"MODEL_DIR calculado: {MODEL_DIR}")
print(f"Caminho COMPLETO esperado para o MODELO: {MODEL_PATH}")
print(f"Verificando se o arquivo do MODELO existe: {os.path.exists(MODEL_PATH)}")
print(f"Caminho COMPLETO esperado para o LIMIAR: {THRESHOLD_PATH}")
print(f"Verificando se o arquivo do LIMIAR existe: {os.path.exists(THRESHOLD_PATH)}")
print(f"--- FIM DEBUG CAMINHOS ---")

try:
    model = joblib.load(MODEL_PATH)
    optimal_threshold = joblib.load(THRESHOLD_PATH) # O valor do limiar (0.61) está dentro deste arquivo
    print(f"Modelo '{MODEL_PATH}' e limiar '{THRESHOLD_PATH}' carregados com sucesso.")
    print(f"Valor do limiar ótimo carregado: {optimal_threshold}")
except FileNotFoundError:
    print(f"ERRO CRÍTICO: Arquivo do modelo ou do limiar NÃO ENCONTRADO após verificação.")
    print(f"Verifique os caminhos exatos e se os arquivos realmente existem nesses locais com os nomes corretos.")
    print(f"- Caminho do modelo que falhou: {MODEL_PATH}")
    print(f"- Caminho do limiar que falhou: {THRESHOLD_PATH}")
    model = None
    optimal_threshold = 0.5
except Exception as e:
    print(f"ERRO ao carregar modelo ou limiar: {e}")
    model = None
    optimal_threshold = 0.5

# --- Funções de Pré-processamento ---
def extract_title_api(name_str):
    """Extrai o título do nome do passageiro."""
    title_search = re.search(' ([A-Za-z]+)\\.', name_str)
    if title_search:
        title = title_search.group(1)
        rare_titles_map = {
            "Capt": "Rare", "Col": "Rare", "Countess": "Rare", "Don": "Rare",
            "Dona": "Rare", "Dr": "Rare", "Jonkheer": "Rare", "Lady": "Rare",
            "Major": "Rare", "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss",
            "Rev": "Rare", "Sir": "Rare"
        }
        return rare_titles_map.get(title, title)
    return "Unknown"

def preprocess_single_input(input_data_dict):
    """
    Pré-processa um único dicionário de dados de entrada para o formato esperado pelo modelo.
    """
    df = pd.DataFrame([input_data_dict])

    df['Title'] = df['Name'].apply(extract_title_api) if 'Name' in df else "Mr"
    df['Age'] = pd.to_numeric(df.get('Age'), errors='coerce').fillna(AGE_MEDIAN)
    df['Embarked'] = df.get('Embarked', EMBARKED_MODE).fillna(EMBARKED_MODE)
    df['Fare'] = pd.to_numeric(df.get('Fare'), errors='coerce').fillna(FARE_MEDIAN)

    df['SibSp'] = pd.to_numeric(df.get('SibSp'), errors='coerce').fillna(0)
    df['Parch'] = pd.to_numeric(df.get('Parch'), errors='coerce').fillna(0)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['HasCabin'] = df['Cabin'].notna().astype(int) if 'Cabin' in df else 0
    
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['FarePerPerson'].replace([np.inf, -np.inf], FARE_MEDIAN, inplace=True)
    df['FarePerPerson'].fillna(FARE_MEDIAN, inplace=True)

    df['Sex'] = df['Sex'].map(SEX_MAP).fillna(SEX_MAP.get('male', 0))
    df['Embarked'] = df['Embarked'].map(EMBARKED_MAP).fillna(EMBARKED_MAP.get(EMBARKED_MODE,0))
    df['Title'] = df['Title'].map(TITLE_MAP).fillna(TITLE_MAP.get('Mr',0))
    df['Pclass'] = pd.to_numeric(df.get('Pclass'), errors='coerce').fillna(3)

    # Garante que todas as 12 features estejam presentes
    for col in MODEL_FEATURES_ORDER:
        if col not in df.columns:
            print(f"Aviso: Coluna '{col}' não encontrada na entrada, usando fallback.")
            if col == 'Age': df[col] = AGE_MEDIAN
            elif col == 'Fare': df[col] = FARE_MEDIAN
            elif col == 'Pclass': df[col] = 3
            elif col == 'SibSp': df[col] = 0
            elif col == 'Parch': df[col] = 0
            elif col == 'Embarked': df[col] = EMBARKED_MAP.get(EMBARKED_MODE,0)
            elif col == 'Sex': df[col] = SEX_MAP.get('male', 0)
            elif col == 'Title': df[col] = TITLE_MAP.get('Mr',0)
            elif col == 'FamilySize': df[col] = 1
            elif col == 'IsAlone': df[col] = 1
            elif col == 'HasCabin': df[col] = 0
            elif col == 'FarePerPerson': df[col] = FARE_MEDIAN
            else: df[col] = 0

    df_processed = df[MODEL_FEATURES_ORDER]
    return df_processed.to_numpy()

# --- Rota da API ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo não carregado. Verifique os logs do servidor.'}), 500

    try:
        input_data_list = request.get_json()

        if not isinstance(input_data_list, list):
            input_data_list = [input_data_list]

        processed_instances = []
        for data_dict in input_data_list:
            if not isinstance(data_dict, dict):
                return jsonify({'error': 'Entrada inválida. Esperado um dicionário ou lista de dicionários.'}), 400
            
            processed_features_np = preprocess_single_input(data_dict)
            processed_instances.append(processed_features_np[0])

        if not processed_instances:
             return jsonify({'error': 'Nenhuma instância válida para processar.'}), 400

        X_processed_batch = np.array(processed_instances)
        probabilities = model.predict_proba(X_processed_batch)[:, 1]
        final_predictions = (probabilities >= optimal_threshold).astype(int)
        
        return jsonify({'predictions': final_predictions.tolist(), 
                        'probabilities_survived': probabilities.tolist(),
                        'threshold_used': optimal_threshold})

    except Exception as e:
        app.logger.error(f"Erro durante a predição: {e}")
        return jsonify({'error': f'Erro ao processar a requisição: {str(e)}'}), 500

if __name__ == '__main__':
    if model is None:
        print("Encerrando: Modelo não pôde ser carregado.")
    else:
        print(f"Iniciando Flask app com modelo: {BEST_MODEL_FILENAME}, limiar carregado: {optimal_threshold}")
        app.run(host='0.0.0.0', port=5000, debug=True)