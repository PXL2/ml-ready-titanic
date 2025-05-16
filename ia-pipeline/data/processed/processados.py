import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import os

def process_data():
    # Obter o caminho absoluto do diretório atual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construir o caminho para o arquivo CSV
    csv_path = os.path.join(os.path.dirname(current_dir), 'raw', 'Titanic-Dataset.csv')

    # Carregar os dados
    df = pd.read_csv(csv_path)
    
    # 1. Tratamento de valores ausentes
    # Extrair título do nome antes de qualquer uso
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Dona'], 'Royalty')
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer')
    df['Title'] = df['Title'].replace(['Don', 'Sir', 'Jonkheer'], 'Sir')
    df['Title'] = df['Title'].replace(['the'], 'Other')

    # Criar feature de tamanho da família antes de usá-la
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Agora sim, preencher valores ausentes de 'Age' usando Title
    df['Age'] = df.groupby(['Pclass', 'Sex', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Preencher Embarked com o valor mais frequente por classe
    df['Embarked'] = df.groupby('Pclass')['Embarked'].transform(lambda x: x.fillna(x.mode()[0]))
    
    # Preencher Fare com a mediana por classe e número de pessoas
    df['Fare'] = df.groupby(['Pclass', 'FamilySize'])['Fare'].transform(lambda x: x.fillna(x.median()))

    # 2. Feature Engineering
    # Agrupar títulos menos comuns
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Mme': 'Mrs',
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    df['Title'] = df['Title'].map(title_mapping)

    # Criar features de família
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['IsSmallFamily'] = ((df['FamilySize'] > 1) & (df['FamilySize'] <= 4)).astype(int)
    df['IsLargeFamily'] = (df['FamilySize'] > 4).astype(int)
    df['HasSpouse'] = (df['SibSp'] > 0).astype(int)
    df['HasChildren'] = (df['Parch'] > 0).astype(int)

    # Criar feature de cabine
    df['Cabin'] = df['Cabin'].fillna('U')
    df['Cabin'] = df['Cabin'].str[0]
    df['HasCabin'] = (df['Cabin'] != 'U').astype(int)
    
    # Criar features de preço
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['FarePerClass'] = df['Fare'] / df['Pclass']
    
    # Criar features de idade
    df['AgeBin'] = pd.qcut(df['Age'], 5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Very Senior'])
    df['IsChild'] = (df['Age'] < 12).astype(int)
    df['IsElderly'] = (df['Age'] > 60).astype(int)
    
    # Criar features de preço
    df['FareBin'] = pd.qcut(df['Fare'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    df['IsExpensive'] = (df['Fare'] > df['Fare'].quantile(0.75)).astype(int)
    
    # Criar interações entre features
    df['ClassSex'] = df['Pclass'].astype(str) + '_' + df['Sex']
    df['ClassTitle'] = df['Pclass'].astype(str) + '_' + df['Title']
    df['AgeClass'] = df['AgeBin'].astype(str) + '_' + df['Pclass'].astype(str)

    # 3. Codificação de variáveis categóricas
    le = LabelEncoder()
    categorical_features = ['Sex', 'Embarked', 'Cabin', 'Title', 'AgeBin', 'FareBin', 
                          'ClassSex', 'ClassTitle', 'AgeClass']
    
    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])

    # 4. Normalização de features numéricas
    scaler = RobustScaler()  # Mais robusto a outliers que StandardScaler
    numeric_features = ['Age', 'Fare', 'FarePerPerson', 'FarePerClass']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # 5. Seleção de features
    features = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
        'Cabin', 'Title', 'FamilySize', 'IsAlone', 'IsSmallFamily', 
        'IsLargeFamily', 'FarePerPerson', 'AgeBin', 'FareBin',
        'HasCabin', 'HasSpouse', 'HasChildren', 'IsChild', 'IsElderly',
        'IsExpensive', 'FarePerClass', 'ClassSex', 'ClassTitle', 'AgeClass'
    ]
    df_processed = df[features + ['Survived']]

    print("Dados processados com sucesso!")
    print("\nShape dos dados processados:")
    print(f"Dataset: {df_processed.shape}")
    
    return df_processed

if __name__ == "__main__":
    process_data()
