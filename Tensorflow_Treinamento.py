# Algoritmo 1: Treinamento do Modelo

# Importando as bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import pandas as pd
import matplotlib.pyplot as plt

# Carregando o conjunto de dados MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizando os dados de entrada
X_train = X_train / 255.0
X_test = X_test / 255.0

# Criando o modelo de rede neural
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes (dígitos de 0 a 9)
])

# Compilando o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinando o modelo
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Salvando o modelo treinado para uso posterior
model.save('modelo_mnist.h5')

# Criando um DataFrame do pandas para as métricas de treinamento
history_df = pd.DataFrame(history.history)

# Plotando as métricas de treinamento usando Pandas e Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gráfico da perda (loss)
history_df[['loss', 'val_loss']].plot(ax=axes[0])
axes[0].set_title('Perda durante o Treinamento e Validação')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Perda')

# Gráfico da acurácia (accuracy)
history_df[['accuracy', 'val_accuracy']].plot(ax=axes[1])
axes[1].set_title('Acurácia durante o Treinamento e Validação')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Acurácia')

plt.show()
