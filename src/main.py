import torch
import numpy as np
import torch.nn as nn
import math
from tensorboardX import SummaryWriter

# Directly create a tensor in torch
a = torch.FloatTensor(3,2)
a.zero_()
a = torch.FloatTensor([[1,2],[3,4],[5,6]])

# Create a tensor from numpy
n = np.zeros(shape =(3,2))
a = torch.tensor(n)

# Specify the type of tensor
#b = a.to('cuda')
c = a.to('cpu')

# Example of gradient
v1 = torch.tensor([1.0,1.0], requires_grad=True)
v2 = torch.tensor([4.0,4.0])
v_sum = v1 + v2
v_res = (v_sum*3).sum()
print("Is Leaf: ", v1.is_leaf, v2.is_leaf, v_sum.is_leaf, v_res.is_leaf)
print("Requires Grad:", v1.requires_grad, v2.requires_grad, v_sum.requires_grad, v_res.requires_grad)
v_res.backward()
print("Gradient: ", v1.grad)

# Simple nn
s = nn.Sequential(
    nn.Linear(2,5),
    nn.ReLU(),
    nn.Linear(5,20),
    nn.ReLU(),
    nn.Linear(20,10),
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1)
)

# Create a nn with sequential layers
class OurModule(nn.Module):
    # create a nn with n inputs, m outputs, and dropout probability
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential(
            # Linear layer with n inputs and 5 outputs
            nn.Linear(num_inputs, 5),
            # ReLU activation function
            nn.ReLU(),
            # Linear layer with 5 inputs and 20 outputs
            nn.Linear(5, 20),
            nn.ReLU(),
            # Linear layer with 20 inputs and m outputs
            nn.Linear(20, num_classes),
            # Dropout layer with probability of dropout_prob
            nn.Dropout(p=dropout_prob),
            # Output layer with Softmax function
            nn.Softmax(dim=1)
        )

    # Forward function override
    def forward(self, x):
        return self.pipe(x)

# if __name__ == "__main__":
#     # Create a nn with 2 inputs and 3 outputs
#     net = OurModule(num_inputs=2, num_classes=3)
#     print(net)

#     # Feed forward
#     v = torch.FloatTensor([[2, 3]])
#     out = net(v)
#     print(out)

#     # Check if cuda is available
#     print("Cuda's availability is %s" % torch.cuda.is_available())
#     if torch.cuda.is_available():
#         print("Data from cuda: %s" % out.to('cuda'))


if __name__ == "__main__":
    writer = SummaryWriter()

    funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan}

    for angle in range(-360, 360):
        angle_rad = angle * math.pi / 180
        for name, fun in funcs.items():
            val = fun(angle_rad)
            writer.add_scalar(name, val, angle)

    writer.close()
