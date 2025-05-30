import requests
import json

# Dados de exemplo de um passageiro do Titanic
passageiro_teste = {
    "Pclass": 3,
    "Name": "Mr. John Doe",
    "Sex": "male",
    "Age": 30,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": "123456",
    "Fare": 7.75,
    "Cabin": None,
    "Embarked": "S"
}

# URL da API
url = "http://localhost:5000/predict"

# Enviando a requisição
try:
    response = requests.post(url, json=passageiro_teste)
    
    # Verificando se a requisição foi bem sucedida
    if response.status_code == 200:
        resultado = response.json()
        print("\nResultado da predição:")
        print(f"Probabilidade de sobrevivência: {resultado['probabilities_survived'][0]:.2%}")
        print(f"Predição final (sobreviveu?): {'Sim' if resultado['predictions'][0] == 1 else 'Não'}")
        print(f"Limiar usado: {resultado['threshold_used']}")
    else:
        print(f"Erro na requisição: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Erro ao conectar com a API: {e}") 