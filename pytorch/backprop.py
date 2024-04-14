import torch


X = torch.tensor(1.0)  # Input
Y = torch.tensor(2.0)  # Output

# Weight
W = torch.tensor(1.0, requires_grad=True)

# Forward pass
Y_hat = W * X
loss = (Y_hat - Y)**2

print(loss)

# Backward pass
loss.backward()

print(W.grad)

# Update weights

# Next forward and backward pass...

# Manual zeroing of the gradients

W.grad.zero_()
print(W.grad)