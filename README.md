# Roteiro 2 - Deeplearning

## Link para o Collab - Tensorflow
https://colab.research.google.com/drive/1QoLKH6jtb8aVYJRHHrLGm2Ckc8XJqUAG?usp=sharing

## Link para o Collab - Torch
[https://colab.research.google.com/drive/1QoLKH6jtb8aVYJRHHrLGm2Ckc8XJqUAG?usp=sharing](https://colab.research.google.com/drive/1c4JuHB-FPEvTF8lte_TGMTfySWgYQd18?usp=sharing)

## Descrição:

Ambos os códigos estão utilizando a base de dados MNIST, um conjunto de 60.000 imagens em tons de cinza de 28x28 dos 10 dígitos, junto com um conjunto de teste de 10.000 imagens. Os códigos serão executados no seguinte formato:

### Script de Treinamento 
1. Importação das bibliotecas.
2. Carregamento dos Dados.
3. Normalização - As imagens são normalizadas dividindo os valores por 255 (para que fiquem entre 0 e 1).
4. Criação do Modelo - Uma rede neural simples é criada com três camadas: uma camada densa com 128 neurônios, outra com 64 neurônios, e uma última camada de saída com 10 neurônios, usando a função de ativação softmax.
5. Treinamento - O modelo é compilado e treinado usando a função de perda de entropia cruzada categórica e o otimizador Adam.
6. Exibição de Métricas - O Pandas é usado para exibir as previsões reais e as preditas em formato de uma plotagem.

### Script de Predição
1. Importação.
2. Carregar os dados de teste.
3. Normalizar os dados.
4. Carregar o modelo salvo.
5. Avaliar o modelo.
6. Fazer Predições.
7. Exibir métricas.
