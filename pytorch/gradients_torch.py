 
#  1.  Design model (input, output size, forward pass)
#  2.  Construct loss and optimizer
#  3.  Training loop
#      - forward pass: compute prediction
#      - backward pass: gradients
#      - update weights


import torch
import torch.nn as nn

# f = w * x

# f = 2 * x

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)

y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 5000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(y, y_pred)

    # gradients == backward pass
    l.backward()  # dl/dw or gradient calculation

    # update weights
    optimizer.step()
        
    # to zero out the gradients
    optimizer.zero_grad()
    if epoch % 100 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

