# Pipeline de IA para Predição de Sobrevivência no Titanic

Este projeto implementa um pipeline completo de Machine Learning, desde o pré-processamento e limpeza de dados do famoso dataset do Titanic até o treinamento, otimização e avaliação de modelos preditivos. O objetivo principal é classificar os passageiros entre sobreviventes e não sobreviventes, com foco em alcançar alta precisão para ambas as classes.

O trabalho original de pré-processamento foi desenvolvido em colaboração por Pedro Lima e Carla Santana, e este repositório expande essa base para um pipeline de IA de ponta a ponta.

## 🚀 Funcionalidades Principais

O pipeline é composto pelas seguintes etapas:

### 1. Pré-processamento e Limpeza de Dados (`ia-pipeline/data/processed/processados.py`)
* **Carregamento de Dados:** Leitura do dataset bruto.
* **Engenharia de Features Inicial:**
    * Extração de `Title` (título) a partir dos nomes dos passageiros e normalização de títulos raros.
* **Tratamento de Valores Ausentes:**
    * `Age`: Imputação pela mediana agrupada por `Pclass`, `Sex` e `Title`.
    * `Embarked`: Imputação pela moda agrupada por `Pclass`.
    * `Fare`: Imputação pela mediana agrupada por `Pclass` e `FamilySize`.
    * `Cabin`: Transformada na feature binária `HasCabin`.
* **Engenharia de Features Adicional:**
    * Criação de `FamilySize` (SibSp + Parch + 1).
    * Criação de `IsAlone` (baseado no `FamilySize`).
    * Criação de `FarePerPerson` (Fare / FamilySize).
* **Codificação de Variáveis Categóricas:**
    * Transformação das features `Sex`, `Embarked` e `Title` em representações numéricas usando `LabelEncoder`.
* **Seleção e Remoção de Colunas:** Remoção de colunas não informativas ou redundantes (`PassengerId`, `Name`, `Ticket`, `Cabin` (original), `SibSp`, `Parch`).
* **Exportação:** Salvamento do dataset pré-processado para a próxima etapa.

### 2. Treinamento e Otimização de Modelos (`ia-pipeline/data/processed/treino/train.py`)
* **Carregamento de Dados Processados.**
* **Divisão dos Dados:** Separação em conjuntos de treino e teste, com estratificação pela variável alvo (`Survived`).
* **Balanceamento de Classes (SMOTE):**
    * Aplicação da técnica SMOTE (Synthetic Minority Over-sampling Technique) *apenas* no conjunto de treinamento para lidar com o desbalanceamento entre sobreviventes e não sobreviventes.

    **Antes do SMOTE:**
    A distribuição original da variável alvo no conjunto de treino mostra um desbalanceamento.

    ![Distribuição da Variável Alvo (y_train) - ANTES do SMOTE](ia-pipeline/assets/imagens/Antes%20do%20SMOTE.png)

    *Figura 1: Distribuição da variável alvo no conjunto de treino antes da aplicação do SMOTE.*


    **Depois do SMOTE:**
    Após a aplicação do SMOTE, as classes no conjunto de treino ficam balanceadas, auxiliando o modelo a aprender de forma mais eficaz.

    ![Distribuição da Variável Alvo (y_train_smote) - DEPOIS do SMOTE](ia-pipeline/assets/imagens/Depois%20do%20SMOTE.png)

    *Figura 2: Distribuição da variável alvo no conjunto de treino após a aplicação do SMOTE.*

* **Pipeline de Modelagem:**
    * Uso de `StandardScaler` para normalizar os dados antes do treinamento.
    * Treinamento de múltiplos algoritmos de classificação: Regressão Logística, SVM, Random Forest, Gradient Boosting e um ensemble Stacking.
* **Otimização de Hiperparâmetros:**
    * Utilização de `GridSearchCV` com validação cruzada estratificada (`StratifiedKFold`) para encontrar a melhor combinação de hiperparâmetros para cada modelo.
    * Métrica de otimização: `precision_macro` para buscar um bom equilíbrio de precisão entre as classes.
    * Implementação de cache para os resultados do `GridSearchCV` para acelerar execuções futuras.
* **Ajuste de Limiar de Decisão (Thresholding):**
    * Desenvolvimento de uma lógica customizada para encontrar o limiar de probabilidade ótimo que maximize a chance de atingir as metas de precisão (>80%) para ambas as classes (sobreviventes e não sobreviventes).
* **Avaliação e Seleção do Melhor Modelo:**
    * Análise detalhada da performance dos modelos usando `classification_report` e `confusion_matrix`.
    * Seleção do modelo final com base no desempenho e nas metas de precisão estabelecidas.
* **Salvamento do Modelo:** O melhor modelo treinado e seu limiar ótimo são salvos em disco (`.joblib`) para futuras predições ou deploy.

## 🛠️ Tecnologias Utilizadas
* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (para SMOTE)
* Matplotlib / Seaborn (para visualizações, embora não geradas diretamente pelos scripts `processados.py` ou `train.py`, foram base para as imagens de SMOTE)
* Joblib (para salvar modelos e cache)

## 📁 Estrutura do Projeto

A estrutura de pastas do projeto está organizada da seguinte forma (considerando a raiz como `PYTHON SUDESTE` localmente):

PYTHON SUDESTE/
├── .gitignore
├── LICENSE
├── README.md
├── ia-pipeline/
│   ├── assets/
│   │   └── imagens/
│   │       ├── Antes do SMOTE.png
│   │       └── Depois do SMOTE.png
│   ├── data/
│   │   ├── processed/
│   │   │   ├── processados.py
│   │   │   ├── processed_titanic_data.csv
│   │   │   └── treino/
│   │   │       └── train.py
│   │   └── raw/
│   │       └── Titanic-Dataset.csv
│   ├── env/                    # Ambiente virtual (sugestão)
│   ├── models/                 # Modelos treinados salvos
│   ├── models_cache/           # Cache do GridSearchCV
│   ├── notebooks/
│   │   └── dados.ipynb         # Notebook original de exploração e desenvolvimento
│   └── src/                    # Código fonte adicional (ex: API)
│       └── api/
│           └── app.py
└── requirements.txt            # Dependências do projeto (sugestão de criação)

## 🚀 Começo Rápido

Siga os passos abaixo para executar o pipeline:

1.  **Clone o Repositório:**
    ```bash
    git clone [URL_DO_SEU_REPOSITORIO_AQUI]
    cd NOME_DA_PASTA_DO_PROJETO_LOCAL # Ex: cd PYTHON SUDESTE
    ```

2.  **Crie e Ative um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv env
    # No Linux/macOS:
    source env/bin/activate
    # No Windows:
    # env\Scripts\activate
    ```

3.  **Instale as Dependências:**
    (Crie um arquivo `requirements.txt` com as bibliotecas listadas em "Tecnologias Utilizadas" se ainda não o fez)
    ```bash
    pip install -r requirements.txt
    ```
    Se não tiver um `requirements.txt`, instale manualmente:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn joblib matplotlib seaborn
    ```

4.  **Execute o Script de Pré-processamento:**
    Este script irá limpar os dados e gerar o `processed_titanic_data.csv`.
    ```bash
    python ia-pipeline/data/processed/processados.py
    ```

5.  **Execute o Script de Treinamento:**
    Este script carregará os dados processados, treinará os modelos e salvará o melhor modelo.
    ```bash
    python ia-pipeline/data/processed/treino/train.py
    ```

## 🎯 Resultados Esperados
O pipeline busca identificar o melhor modelo de classificação que atinja uma **precisão de pelo menos 80% tanto para prever passageiros que sobreviveram quanto para os que não sobreviveram**, após ajuste fino do limiar de decisão. Os resultados detalhados e o modelo selecionado são exibidos ao final da execução do script de treinamento.
