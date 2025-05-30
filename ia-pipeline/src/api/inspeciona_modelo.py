import joblib

modelo = joblib.load('ia-pipeline/models/best_titanic_model_plots_Logistic_Regression.joblib')
print(modelo)
try:
    print("\nFeatures esperadas:", modelo.feature_names_in_)
except AttributeError:
    print("Atributo feature_names_in_ não encontrado.")
    # Tente acessar steps do pipeline, se for um Pipeline
    try:
        print("Steps do pipeline:", modelo.named_steps)
    except AttributeError:
        print("Não é um pipeline scikit-learn.") 