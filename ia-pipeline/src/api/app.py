from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Caminho para o modelo treinado
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'processed', 'best_model.joblib')

# Carregar o modelo
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Espera receber um dicionário com os dados das features
    if isinstance(data, dict):
        X = np.array([list(data.values())])
    elif isinstance(data, list):
        X = np.array([list(d.values()) for d in data])
    else:
        return jsonify({'error': 'Formato de entrada inválido'}), 400
    prediction = model.predict(X)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 