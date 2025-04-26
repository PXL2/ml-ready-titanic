# Pré-processamento do Dataset Titanic

## 📌 Visão Geral
Este projeto realiza uma análise exploratória completa e pré-processamento do dataset Titanic, preparando os dados para modelos preditivos de machine learning. O trabalho foi desenvolvido em colaboração por [Pedro Lima](https://github.com/PXL2) e [Carla Santana](https://github.com/carlaasantana).

## 🔍 Principais Funcionalidades

## 🛠️ Funcionalidades
- **Limpeza de Dados**:
  - Tratamento de valores faltantes em Age, Embarked e Cabin
  - Remoção de outliers na tarifa (Fare)
- **Engenharia de Features**:
  - Criação de `Tamanho_Familia` a partir de SibSp e Parch
  - Adição da feature binária `Sozinho`
  - Extração de `Titulo` dos nomes dos passageiros
- **Codificação**:
  - Transformação de variáveis categóricas (Sexo, Embarked, Pclass, Titulo) em dummy variables

## 📂 Estrutura do Projeto
titanic-preprocess/
├── dados/
│ ├── brutos/ # Dataset original
│ └── processados/ # Dados limpos
├── notebooks/ # Jupyter notebooks
├── src/ # Scripts Python
├── README.md
└── requirements.txt

## 🚀 Começo Rápido
1. Clone o repositório:
```bash
git clone https://github.com/PXL2/Python-sudeste.git
cd titanic-preprocess
pip install -r requirements.txt
python src/preprocess.py
```

