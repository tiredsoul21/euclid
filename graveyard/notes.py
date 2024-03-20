import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from torchviz import make_dot # 0.0.2

a = torch.zeros(3, 4)  # 3x4 tensor of zeros rows x columns
print(a.dtype)         # torch.float32 by default
a[1,2] = 1                  # 1x2 element is now 1
a[0] = 2                    # the entire first row is now 2
print(a[0,:])               # first row, all columns
print(a[:,2])               # all rows, second column
print(torch.sum(a, dim=0))  # sum of each column
print(torch.sum(a, dim=1))  # sum of each row
print(torch.sum(a))         # sum of all elements
 # sum of each row, keeping the dimensions (3x1)
print(torch.sum(a, dim=1, keepdim=True))

# Sample from a multinomial distribution with deterministic random number generator
gen = torch.Generator().manual_seed(42)
b = torch.multinomial(torch.tensor([0.2, 0.3, 0.5]), 4, \
                      replacement=True, generator=gen)
torch.randn(3, 4, generator=gen)  # Random normal distribution
print(b)

c = torch.tensor([1, 1, 2, 3]) # Create a tensor from a list

# One-hot encode the tensor (creates a matrix with 3 rows and 4 columns)
# with the column filled with zeros except for the ith element
d = F.one_hot(c, num_classes=4).float()
print(d)

e = a @ d  # Matrix multiplication

print(torch.arange(6))  # 1D tensor with values 0 to 5

# logits are the raw scores output by the last layer of the model -- "log counts"

# # Plot the tensor
# a = torch.tensor([[1, 2, 3, 4],
#                   [5, 6, 7, 8],
#                   [9, 10, 11, 12]])
# plt.imshow(a, cmap='gray', interpolation='none')
# plt.show()

f = torch.tensor([1., 2., 3., 4.], requires_grad=True)
f.grad = None  # Zero the gradients
g = f * 2
h = g.sum()
i = torch.tensor([0.0, 1.1, 2.2, 3.3])
h.backward()  # Compute the gradients

print(f.grad)  # df/dx = 2
print(f.requires_grad) # True - leaf variable
print(g.requires_grad) # True - not a leaf variable
print(h.requires_grad) # True - not a leaf variable
print(i.requires_grad) # False - leaf variable

# make 2 3x3x4 tensors
aa = torch.randn(2, 4, 4)
bb = torch.randn(2, 4, 3)
# concatenate them along the third dimension - naively
cc = torch.cat([aa, bb], dim=2)
print(cc.shape)  # torch.Size([2, 4, 7])
print(cc.view(8, 7).shape) # Reshape to 8x7 -- flatten the first two dimensions
print(cc)
print(cc.view(8, 7))
print(cc[:, torch.arange(2), :])  # Select the first two rows of each 4x7 matrix
random.shuffle(cc)  # Shuffle the first dimension

# To determine a good learning rate. Plot the loss as a function of the learning rate

# Display the graph
make_dot(h, params=dict(f=f))

# A way to see dead neurons in a neural network (from neurons pushed into flat regions)
plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')
