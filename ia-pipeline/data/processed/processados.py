# ia-pipeline/data/processed/processados.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Definir os caminhos baseados na localização do script
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # /ia-pipeline/data/processed
RAW_DATA_PATH = os.path.join(BASE_DIR, '..', 'raw', 'Titanic-Dataset.csv') # /ia-pipeline/data/raw/Titanic-Dataset.csv
PROCESSED_DATA_OUTPUT_PATH = os.path.join(BASE_DIR, 'processed_titanic_data.csv') # /ia-pipeline/data/processed/processed_titanic_data.csv

def load_data(path):
    """Carrega os dados de um arquivo CSV."""
    print(f"Carregando dados de: {path}")
    return pd.read_csv(path)

def extract_title(df):
    """Extrai o título do nome do passageiro."""
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    # Agrupar títulos raros
    rare_titles = {
        "Capt": "Rare", "Col": "Rare", "Countess": "Rare", "Don": "Rare",
        "Dona": "Rare", "Dr": "Rare", "Jonkheer": "Rare", "Lady": "Rare",
        "Major": "Rare", "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss",
        "Rev": "Rare", "Sir": "Rare"
    }
    df['Title'] = df['Title'].replace(rare_titles)
    return df

def preprocess_data(df):
    """Aplica todas as etapas de pré-processamento e engenharia de features."""
    print("Iniciando pré-processamento...")

    df_processed = df.copy()

    # 1. Extrair Título
    df_processed = extract_title(df_processed)
    print("Feature 'Title' criada e normalizada.")

    # 2. Tratar 'Age' usando a mediana agrupada por Pclass, Sex, e Title
    # Primeiro, certifique-se que os agrupadores não têm NaN para evitar erros no groupby
    cols_for_grouping_age = ['Pclass', 'Sex', 'Title']
    for col in cols_for_grouping_age:
        if df_processed[col].isnull().any():
            print(f"Alerta: Valores nulos encontrados em '{col}' antes de agrupar para 'Age'. Tratando com a moda.")
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    df_processed['Age'] = df_processed.groupby(cols_for_grouping_age)['Age'].transform(lambda x: x.fillna(x.median()))
    # Caso ainda haja NaNs em Age (ex: grupos sem nenhuma idade não nula), preencher com a mediana global
    if df_processed['Age'].isnull().any():
        print("Preenchendo NaNs restantes em 'Age' com a mediana global.")
        df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
    print("Valores ausentes em 'Age' tratados.")

    # 3. Tratar 'Embarked'
    # Primeiro, certifique-se que Pclass não tem NaN
    if df_processed['Pclass'].isnull().any():
         print(f"Alerta: Valores nulos encontrados em 'Pclass' antes de agrupar para 'Embarked'. Tratando com a moda.")
         df_processed['Pclass'] = df_processed['Pclass'].fillna(df_processed['Pclass'].mode()[0])

    df_processed['Embarked'] = df_processed.groupby('Pclass')['Embarked'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'S'))
    if df_processed['Embarked'].isnull().any(): # Se ainda houver NaNs
        df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
    print("Valores ausentes em 'Embarked' tratados.")

    # 4. Tratar 'Fare'
    # Criar FamilySize temporariamente para imputar Fare, se necessário (se FamilySize ainda não existe)
    if 'FamilySize' not in df_processed.columns:
        df_processed['TEMP_FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        cols_for_grouping_fare = ['Pclass', 'TEMP_FamilySize']
    else:
        cols_for_grouping_fare = ['Pclass', 'FamilySize']

    for col in cols_for_grouping_fare:
        if df_processed[col].isnull().any():
            print(f"Alerta: Valores nulos encontrados em '{col}' antes de agrupar para 'Fare'. Tratando com a moda/mediana.")
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    df_processed['Fare'] = df_processed.groupby(cols_for_grouping_fare)['Fare'].transform(lambda x: x.fillna(x.median()))
    if df_processed['Fare'].isnull().any():
        df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
    print("Valores ausentes em 'Fare' tratados.")
    if 'TEMP_FamilySize' in df_processed.columns:
        df_processed.drop('TEMP_FamilySize', axis=1, inplace=True)


    # 5. Engenharia de Features Adicionais
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
    df_processed['HasCabin'] = df_processed['Cabin'].notna().astype(int)
    # Evitar divisão por zero para FarePerPerson
    df_processed['FarePerPerson'] = df_processed['Fare'] / df_processed['FamilySize']
    df_processed['FarePerPerson'].replace([np.inf, -np.inf], 0, inplace=True) # Lida com FamilySize == 0 (se ocorrer)
    df_processed['FarePerPerson'].fillna(0, inplace=True) # Lida com NaN resultante de Fare=0/FamilySize=0
    print("Features 'FamilySize', 'IsAlone', 'HasCabin', 'FarePerPerson' criadas.")

    # 6. Encoding de Variáveis Categóricas
    label_encoders = {}
    for column in ['Sex', 'Embarked', 'Title']:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column])
        label_encoders[column] = le # Poderia ser salvo se necessário para decodificar depois
    print("Variáveis categóricas ('Sex', 'Embarked', 'Title') codificadas.")

    # 7. Selecionar e Remover Colunas
    # 'Ticket' e 'Cabin' são complexas e foram simplificadas ou removidas. 'Name' e 'PassengerId' não são preditivas.
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    df_processed.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Colunas descartadas: {cols_to_drop}")

    # Garantir que a coluna 'Survived' (alvo) esteja presente, se não, não fazer nada
    if 'Survived' not in df_processed.columns and 'Survived' in df.columns:
         df_processed['Survived'] = df['Survived']


    # Verificar se ainda existem NaNs e preencher com uma estratégia genérica (mediana/moda)
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            print(f"Aviso: Coluna '{col}' ainda contém {df_processed[col].isnull().sum()} NaNs após pré-processamento. Preenchendo...")
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
            else:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

    print("Pré-processamento concluído.")
    return df_processed

def save_processed_data(df, path):
    """Salva o DataFrame processado em um arquivo CSV."""
    try:
        df.to_csv(path, index=False)
        print(f"Dados processados salvos em: {path}")
    except Exception as e:
        print(f"Erro ao salvar dados processados: {e}")


if __name__ == '__main__':
    # Carregar dados brutos
    raw_df = load_data(RAW_DATA_PATH)

    # Aplicar pré-processamento
    processed_df = preprocess_data(raw_df)

    # Salvar dados processados
    save_processed_data(processed_df, PROCESSED_DATA_OUTPUT_PATH)

    print("\nVisualização das primeiras linhas dos dados processados:")
    print(processed_df.head())
    print("\nInformações dos dados processados:")
    processed_df.info()
    print("\nEstatísticas descritivas dos dados processados:")
    print(processed_df.describe())
    print("\nVerificação de valores nulos nos dados processados:")
    print(processed_df.isnull().sum())