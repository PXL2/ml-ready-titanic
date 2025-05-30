# Pré-processamento do Dataset Titanic

Este projeto realiza uma análise exploratória completa e pré-processamento do dataset Titanic, preparando os dados para modelos preditivos de machine learning. O trabalho foi desenvolvido em colaboração por Pedro Lima e Carla Santana.

## ✨ Principais Funcionalidades

Aqui estão detalhadas as principais transformações e técnicas aplicadas aos dados:

### 🧹 Limpeza de Dados
* Tratamento de valores faltantes em `Age`, `Embarked` e `Cabin`.
* Remoção de outliers na tarifa (`Fare`). *(Nota: Esta etapa pode precisar ser verificada em relação aos scripts que geramos, que focaram mais na imputação robusta).*

### 🛠️ Engenharia de Features
* Criação da feature `Tamanho_Familia` a partir de `SibSp` e `Parch`.
* Adição da feature binária `Sozinho` para indicar passageiros viajando sozinhos.
* Extração e normalização da feature `Titulo` a partir dos nomes dos passageiros.

### ⚖️ Balanceamento de Classes
Para lidar com o desbalanceamento na variável alvo (`Survived`), utilizamos a técnica SMOTE (Synthetic Minority Over-sampling Technique) no conjunto de treinamento. Isso ajuda a previnir que o modelo de machine learning seja enviesado em direção à classe majoritária.

**Antes do SMOTE:**
A distribuição original da variável alvo no conjunto de treino mostra um desbalanceamento entre as classes "Não Sobreviveu" e "Sobreviveu".

![Distribuição da Variável Alvo (y_train) - ANTES do SMOTE](assets/images/distribuicao_antes_smote.png)
*Figura 1: Distribuição da variável alvo antes da aplicação do SMOTE.*

**Depois do SMOTE:**
Após a aplicação do SMOTE, as classes no conjunto de treino ficam balanceadas.

![Distribuição da Variável Alvo (y_train_smote) - DEPOIS do SMOTE](assets/images/distribuicao_depois_smote.png)
*Figura 2: Distribuição da variável alvo após a aplicação do SMOTE.*

### 🔡 Codificação
* Transformação de variáveis categóricas (`Sex`, `Embarked`, `Pclass`, `Title`) em formato numérico adequado para os algoritmos de machine learning. *(Nos scripts que geramos, usamos principalmente LabelEncoding seguido de StandardScaler no pipeline de treino).*

## 📁 Estrutura do Projeto
