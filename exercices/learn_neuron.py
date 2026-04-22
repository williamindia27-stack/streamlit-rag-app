import torch
import torch.nn as nn
import torch.optim as optim

# training data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# one neuron
model = nn.Linear(1, 1)

# loss function
loss_fn = nn.MSELoss()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# training loop
for epoch in range(1000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# show learned values
print("Weight:", model.weight.item())
print("Bias:", model.bias.item())  

# new input
new_x = torch.tensor([[5.0]])

# prediction
prediction = model(new_x)

print("Prediction for x=5:", prediction.item())