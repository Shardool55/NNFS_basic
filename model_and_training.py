# Ideal Pytorch Pipeline - Model, Training and Evaluation

import torch
from data_preprocessing import X_train, X_test, y_train, y_test

# Numpy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

print(X_train_tensor.shape)
print(y_train_tensor.shape)

# Defining the model
class MySimpleNN():
    def __init__(self, X):
        self.weights = torch.rand(X.shape[1], 1, dtype=torch.float64, requires_grad=True)   #initializing weights wwith random values
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    
    def forward(self, X):
        z = torch.matmul(X, self.weights) + self.bias     # X is input tensor  // Z=WX+B
        y_pred = torch.sigmoid(z)
        return y_pred

    def loss_function(self, y_pred, y):
        #clamp predictions to avaoid log(0)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1-epsilon)

        #calculating loss (cross entropy)
        loss = -(y_train_tensor*torch.log(y_pred) + (1-y_train_tensor)*torch.log(1-y_pred)).mean()  # mean helps to find the average loss
        return loss

# Marking important parameters
learning_rate = 0.1
epochs = 100

# Training Pipeline
#create model
model = MySimpleNN(X_train_tensor) #Id we do model.weights, we see matrix of 30,1 and model.weigh as 0
#define loops for the number of epochs defined
for epoch in range(epochs):
    #forward pass
    y_pred = model.forward(X_train_tensor)
    #Loss calculation
    loss = model.loss_function(y_pred, y_train_tensor)
    #Backward pass
    loss.backward()
    #updating parameters
    with torch.no_grad():
        model.weights -= learning_rate*model.weights.grad
        model.bias -= learning_rate*model.bias.grad

    #zero gradients
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    #print loss in each epoch
    print(f'Epoch:{epoch+1}, Loss: {loss.item()}')

print(model.bias)

# Evaluation
#model evaluation
with torch.no_grad():
    y_pred = model.forward(X_test_tensor)
    #scaling the predicted values to be within 0 and 1 with a threshold value
    y_pred = (y_pred > 0.5).float()

print(y_pred)

#accuracy
accuracy = (y_pred == y_test_tensor).float().mean()
print(f"Accuracy: {accuracy.item()}")

# The end accuracy is really poor but our end goal with this notebook was to train a neural network from scratch and gain an intution before moving forward to modern methods of developing neural networks

