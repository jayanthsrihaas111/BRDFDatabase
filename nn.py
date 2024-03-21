import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.w1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
        self.w2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
        self.w3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0) * 1

    def forward(self, x):
        self.z1 = np.dot(x, self.w1)
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.w3)
        return self.z3

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backprop(self, x, y_true, learning_rate):
        y_pred = self.forward(x)
        loss = self.compute_loss(y_pred, y_true)

        d_loss = 2 * (y_pred - y_true) / y_true.size
        d_w3 = np.dot(self.a2.T, d_loss)
        d_loss = np.dot(d_loss, self.w3.T) * self.relu_derivative(self.z2)
        d_w2 = np.dot(self.a1.T, d_loss)
        d_loss = np.dot(d_loss, self.w2.T) * self.relu_derivative(self.z1)
        d_w1 = np.dot(x.T, d_loss)

        # Gradient clipping
        d_w1 = np.clip(d_w1, -1, 1)
        d_w2 = np.clip(d_w2, -1, 1)
        d_w3 = np.clip(d_w3, -1, 1)

        self.w1 -= learning_rate * d_w1
        self.w2 -= learning_rate * d_w2
        self.w3 -= learning_rate * d_w3

        return loss

    def train(self, x_train, y_train, x_val, y_val, epochs, learning_rate, batch_size):
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                batch_loss = self.backprop(x_batch, y_batch, learning_rate)
                epoch_loss += batch_loss
            
            training_losses.append(epoch_loss / len(x_train))

            # Validation
            val_predictions = []
            for i in range(0, len(x_val), batch_size):
                x_batch_val = x_val[i:i + batch_size]
                y_batch_pred = self.forward(x_batch_val)
                val_predictions.append(y_batch_pred)

            y_pred_val = np.vstack(val_predictions)
            val_loss = self.compute_loss(y_pred_val, y_val)
            validation_losses.append(val_loss)

            print(f'Epoch {epoch+1}, Training Loss: {epoch_loss / len(x_train)}, Validation Loss: {val_loss}')

        return training_losses, validation_losses

def evaluate_model(model, X, y, batch_size):
    predictions = []
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i + batch_size]
        y_batch_pred = model.forward(x_batch)
        predictions.append(y_batch_pred)
    y_pred = np.vstack(predictions)
    return model.compute_loss(y_pred, y)

# Load and preprocess the dataset

data = pd.read_csv('/Users/jayanthsrihaas111/Desktop/BRDFDatabase/output.csv')  

X = data.iloc[:, :-3]  # Features
y = data.iloc[:, -3:]  # RGB targets

# Split the dataset into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Neural network parameters
input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 64
output_size = y_train.shape[1]
epochs = 20
learning_rate = 0.01
batch_size = 100  

# Initialize the neural network
nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

# Convert data to numpy arrays for processing
X_train_values = X_train.values
y_train_values = y_train.values
X_val_values = X_val.values
y_val_values = y_val.values
X_test_values = X_test.values
y_test_values = y_test.values

# Train the network
training_losses, validation_losses = nn.train(X_train_values, y_train_values, X_val_values, y_val_values, epochs, learning_rate, batch_size)

# Evaluate the model on the test set
test_loss = evaluate_model(nn, X_test_values, y_test_values, batch_size)
print(f"Test Loss: {test_loss}")

# After training, predict on the test set and compare to actual values
test_predictions = nn.forward(X_test_values)
# Concatenate actual and predicted values for comparison
actual_vs_predicted = np.hstack((y_test_values, test_predictions))
print("Actual vs Predicted RGB values:")
print(actual_vs_predicted)

# Plot training, validation, and test loss
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='-', label='Test Loss')
plt.title('Training and testing errors for epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
#plt.savefig('training_validation_loss.png')
plt.close()