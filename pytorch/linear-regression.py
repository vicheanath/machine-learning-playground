# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#     - forward pass: compute prediction
#     - backward pass: gradients
#     - update weights
#

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32)) # 100x1

y = torch.from_numpy(y_numpy.astype(np.float32)) # 100x1

y = y.view(y.shape[0], 1) # view is similar to reshape in numpy

n_samples, n_features = X.shape

# 1) Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) Loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # Backward pass
    loss.backward()

    # Update
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


# Plot
predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
