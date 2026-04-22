import torch
import torch.nn as nn
import torch.optim as optim

# training data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_true = torch.tensor([[1.0], [4.0], [9.0], [16.0], [25.0]])

# small neural network
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# training
for epoch in range(3000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# prediction
new_x = torch.tensor([[6.0]])
prediction = model(new_x)

print("Prediction for x=6:", prediction.item())