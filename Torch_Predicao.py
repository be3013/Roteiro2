# Algoritmo 2: Predição com o Modelo Treinado em PyTorch e Exibição de Métricas

# Importando as bibliotecas necessárias
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

# Definindo a transformação para normalizar os dados
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Carregando o conjunto de dados MNIST
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definindo a rede neural (mesma arquitetura usada para treinar)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Criando o modelo
model = NeuralNet()

# Carregando os pesos do modelo treinado
model.load_state_dict(torch.load('modelo_mnist_pytorch.pth'))

# Avaliando o modelo e coletando previsões
def evaluate_model(model, test_loader):
    model.eval()  # Colocar o modelo em modo de avaliação
    y_pred = []
    y_real = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_real.extend(labels.numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return np.array(y_real), np.array(y_pred), accuracy

# Fazendo predições e avaliando o modelo
y_test, y_pred, test_accuracy = evaluate_model(model, test_loader)

# Criando um DataFrame do pandas para exibir as métricas de previsão
metrics_df = pd.DataFrame({
    'Real': y_test,
    'Predito': y_pred
})

# Exibindo as primeiras 10 previsões
print(metrics_df.head(10))

# Exibindo a acurácia final
print(f"Acurácia no conjunto de teste: {test_accuracy:.4f}")
