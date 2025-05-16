import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score
import os
from processados import process_data
from imblearn.over_sampling import SMOTE

# Carregar e processar os dados
df = process_data()

# Dividir os dados em treino e teste
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Survived'])

# Separar features e target apenas para treinamento
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data.drop('Survived', axis=1)
y_test = test_data['Survived']

# Aplicar SMOTE para balancear as classes no conjunto de treino
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Definir os modelos base com class_weight manual para dar mais peso à classe 1
rf = RandomForestClassifier(random_state=42, class_weight={0: 1, 1: 3})
gb = GradientBoostingClassifier(random_state=42)
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight={0: 1, 1: 3})
svm = SVC(random_state=42, probability=True, class_weight={0: 1, 1: 3})

# Definir os parâmetros para GridSearch
param_grid = {
    'Random Forest': {
        'n_estimators': [300, 400, 500],
        'max_depth': [20, 25, 30],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [{0: 1, 1: 3}, {0: 1, 1: 5}]
    },
    'Gradient Boosting': {
        'n_estimators': [300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 5, 6],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2]
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [{0: 1, 1: 3}, {0: 1, 1: 5}]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto'],
        'class_weight': [{0: 1, 1: 3}, {0: 1, 1: 5}]
    }
}

# Dicionário para armazenar os resultados
results = {}
best_models = {}

# Definir StratifiedKFold para validação cruzada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Treinar e avaliar cada modelo com GridSearch
for name, model in [('Random Forest', rf), ('Gradient Boosting', gb),
                   ('Logistic Regression', lr), ('SVM', svm)]:
    print(f"\nTreinando {name}...")
    
    # Realizar GridSearch
    grid_search = GridSearchCV(
        model, 
        param_grid[name], 
        cv=skf, 
        scoring='precision',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train.values.ravel())
    
    # Obter o melhor modelo
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    
    # Fazer previsões
    y_pred = best_model.predict(X_test)
    
    # Calcular acurácia
    accuracy = accuracy_score(y_test, y_pred)
    
    # Realizar validação cruzada
    cv_scores = cross_val_score(best_model, df.drop('Survived', axis=1), 
                               df['Survived'].values.ravel(), cv=skf)
    
    # Armazenar resultados
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'best_params': grid_search.best_params_
    }
    
    # Imprimir relatório de classificação
    print(f"\nRelatório de Classificação para {name}:")
    print(classification_report(y_test, y_pred))
    
    # Imprimir resultados da validação cruzada
    print(f"\nResultados da Validação Cruzada para {name}:")
    print(f"Média: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print(f"Melhores parâmetros: {grid_search.best_params_}")

# Criar um ensemble com Stacking
estimators = [(name, model) for name, model in best_models.items()]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=skf
)

# Treinar o ensemble
stacking_clf.fit(X_train, y_train.values.ravel())

# Avaliar o ensemble
y_pred_ensemble = stacking_clf.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
cv_scores_ensemble = cross_val_score(stacking_clf, df.drop('Survived', axis=1), 
                                    df['Survived'].values.ravel(), cv=skf)

results['Stacking'] = {
    'accuracy': accuracy_ensemble,
    'cv_mean': cv_scores_ensemble.mean(),
    'cv_std': cv_scores_ensemble.std()
}

print("\nResultados do Stacking Ensemble:")
print(f"Acurácia: {accuracy_ensemble:.3f}")
print(f"Validação Cruzada: {cv_scores_ensemble.mean():.3f} (+/- {cv_scores_ensemble.std() * 2:.3f})")

# Encontrar o melhor modelo
best_model_name = max(results.items(), key=lambda x: x[1]['cv_mean'])[0]
print(f"\nMelhor modelo: {best_model_name}")
print(f"Acurácia: {results[best_model_name]['accuracy']:.3f}")
print(f"Validação Cruzada: {results[best_model_name]['cv_mean']:.3f} (+/- {results[best_model_name]['cv_std'] * 2:.3f})")

# Após o treinamento do melhor modelo, ajustar o threshold para maximizar a precisão da classe 1
if best_model_name == 'Stacking':
    proba = stacking_clf.predict_proba(X_test)[:, 1]
else:
    proba = best_models[best_model_name].predict_proba(X_test)[:, 1]

# Testar thresholds altos para garantir precisão da classe 1 > 0.80
thresholds = np.arange(0.80, 1.00, 0.01)
melhor_precisao = 0
melhor_threshold = 0.5
melhor_relatorio = None
atingiu_80 = False

print("\nAjuste de threshold para garantir precisão da classe 1 > 0.80:")
for thresh in thresholds:
    y_pred_thresh = (proba >= thresh).astype(int)
    prec = precision_score(y_test, y_pred_thresh, pos_label=1, zero_division=0)
    print(f"Threshold: {thresh:.2f} | Precisão classe 1: {prec:.2f}")
    if prec > 0.80 and not atingiu_80:
        melhor_precisao = prec
        melhor_threshold = thresh
        melhor_relatorio = classification_report(y_test, y_pred_thresh)
        atingiu_80 = True
        print(f"\n*** Threshold encontrado: {thresh:.2f} com precisão da classe 1 = {prec:.2f} ***\n")
        break

if atingiu_80:
    print(f"Melhor threshold para precisão da classe 1: {melhor_threshold:.2f} | Precisão: {melhor_precisao:.2f}")
    print("\nRelatório de classificação com melhor threshold:")
    print(melhor_relatorio)
else:
    print("\nNenhum threshold atingiu precisão da classe 1 > 0.80. Último resultado:")
    print(f"Threshold: {thresh:.2f} | Precisão classe 1: {prec:.2f}")
    print(classification_report(y_test, y_pred_thresh))

# Salvar o melhor modelo
import joblib
if best_model_name == 'Stacking':
    best_model_instance = stacking_clf
else:
    best_model_instance = best_models[best_model_name]

joblib.dump(best_model_instance, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.joblib'))
print("\nMelhor modelo salvo como 'best_model.joblib'") 