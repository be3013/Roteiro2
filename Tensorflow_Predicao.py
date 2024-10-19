# Algoritmo 2: Predição com o Modelo Treinado e Exibição de Métricas

# Importando as bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np

# Carregando o conjunto de dados MNIST
(_, _), (X_test, y_test) = mnist.load_data()

# Normalizando os dados de entrada
X_test = X_test / 255.0

# Carregando o modelo treinado
model = tf.keras.models.load_model('modelo_mnist.h5')

# Avaliando o modelo nos dados de teste
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Fazendo previsões
y_pred = np.argmax(model.predict(X_test), axis=1)

# Criando um DataFrame do pandas para exibir as métricas de previsão
metrics_df = pd.DataFrame({
    'Real': y_test,
    'Predito': y_pred
})

# Exibindo as primeiras 10 previsões
print(metrics_df.head(10))

# Exibindo a acurácia final
print(f"Acurácia no conjunto de teste: {test_accuracy:.4f}")
