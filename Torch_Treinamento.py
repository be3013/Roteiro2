# Algoritmo 1: Treinamento do Modelo com PyTorch

# Importando as bibliotecas necessárias
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt

# Definindo a transformação para normalizar os dados
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Carregando o conjunto de dados MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definindo a rede neural
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

# Definindo a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Função para treinar o modelo
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Estatísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        print(f'Época {epoch+1}/{epochs}, Perda: {epoch_loss:.4f}, Acurácia: {epoch_acc:.4f}')
    
    return history

# Treinando o modelo
history = train_model(model, train_loader, criterion, optimizer)

# Salvando o modelo treinado
torch.save(model.state_dict(), 'modelo_mnist_pytorch.pth')

# Criando um DataFrame do pandas para as métricas de treinamento
history_df = pd.DataFrame(history)

# Plotando as métricas de treinamento usando Pandas e Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gráfico da perda (loss)
history_df[['loss']].plot(ax=axes[0])
axes[0].set_title('Perda durante o Treinamento')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Perda')

# Gráfico da acurácia (accuracy)
history_df[['accuracy']].plot(ax=axes[1])
axes[1].set_title('Acurácia durante o Treinamento')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Acurácia')

plt.show()
