import torch
import torch.nn as nn
import torch.optim as optim

# Example Dataset
X = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                  [0.5, 0.6, 0.7, 0.8],
                  [0.9, 1.0, 1.1, 1.2],
                  [1.3, 1.4, 1.5, 1.6],
                  [1.7, 1.8, 1.9, 2.0]], dtype=torch.float32)

y = torch.tensor([[0.2, 0.3, 0.4],
                  [0.5, 0.6, 0.7],
                  [0.8, 0.9, 1.0],
                  [1.1, 1.2, 1.3],
                  [1.4, 1.5, 1.6]], dtype=torch.float32)

# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 5)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(5, 5)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(5, 3)  # Second hidden layer to output layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = NeuralNetwork()

# Define Loss Function
criterion = nn.MSELoss()  # Mean Squared Error for regression

# Define the SGD Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Basic stochastic gradient descent

# Training Loop
epochs = 100
batch_size = 2  # Mini-batch size for stochastic gradient descent
n_samples = X.size(0)

for epoch in range(epochs):
    permutation = torch.randperm(n_samples)  # Shuffle data at each epoch

    for i in range(0, n_samples, batch_size):
        # Create mini-batches
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X[indices], y[indices]

        # Forward pass
        predictions = model(batch_X)

        # Compute loss
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()  # Reset gradients
        loss.backward()        # Backpropagation

        # Update weights
        optimizer.step()       # Apply gradients

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

print("Training complete!")
