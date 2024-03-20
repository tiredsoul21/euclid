import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

BLOCK_SIZE = 3 # context length: how many characters do we take to predict the next one?
N_EMBD = 10 # the dimensionality of the character embedding vectors
N_HIDDEN = 100 # the number of neurons in the hidden layer of the MLP
MAX_STEPS = 60000 #200000
BATCH_SIZE = 32

class Linear:
    """A simple dense layer with fan-in initialization."""
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
        self.training = True
        self.out = None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        """ Returns layer parameters """
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    """ Batch Normalization for linear layer """
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        self.out = None

    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        """ Returns the model parameters """
        return [self.gamma, self.beta]

class Tanh:
    """ Hyperbolic tangent activation function """
    def __init__(self):
        self.out = None
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        """ No parameters """
        return []


words = open('/home/derrick/repo/euclid/data/names.txt', 'r', encoding='utf-8').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

def build_dataset(words):  
    """ Build the dataset """
    X, Y = [], []
    for w in words:
        context = [0] * BLOCK_SIZE
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((vocab_size, N_EMBD),            generator=g)
layers = [
  Linear(N_EMBD * BLOCK_SIZE, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
  Linear(           N_HIDDEN, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
  Linear(           N_HIDDEN, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
  Linear(           N_HIDDEN, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
  Linear(           N_HIDDEN, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
  Linear(           N_HIDDEN, vocab_size, bias=False), BatchNorm1d(vocab_size),
]
# layers = [
#   Linear(N_EMBD * BLOCK_SIZE, N_HIDDEN), Tanh(),
#   Linear(           N_HIDDEN, N_HIDDEN), Tanh(),
#   Linear(           N_HIDDEN, N_HIDDEN), Tanh(),
#   Linear(           N_HIDDEN, N_HIDDEN), Tanh(),
#   Linear(           N_HIDDEN, N_HIDDEN), Tanh(),
#   Linear(           N_HIDDEN, vocab_size),
# ]

with torch.no_grad():
    # last layer: make less confident
    layers[-1].gamma *= 0.1
    #layers[-1].weight *= 0.1
    # all other layers: apply gain
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 1.0 #5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
    p.requires_grad = True

# same optimization as last time
lossi = []
ud = []

for i in range(MAX_STEPS):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

    # forward pass
    emb = C[Xb] # embed the characters into vectors
    x = emb.view(emb.shape[0], -1) # concatenate the vectors
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb) # loss function

    # backward pass
    for layer in layers:
        layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{MAX_STEPS:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

    # if i >= 1000:
    #     break # AFTER_DEBUG: would take out obviously to run full optimization

# # visualize histograms
# plt.figure(figsize=(20, 4)) # width and height of the plot
# legends = []
# for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
#   if isinstance(layer, Tanh):
#     t = layer.out
#     print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
#     hy, hx = torch.histogram(t, density=True)
#     plt.plot(hx[:-1].detach(), hy.detach())
#     legends.append(f'layer {i} ({layer.__class__.__name__}')
# plt.legend(legends);
# plt.title('activation distribution')
        
# # visualize histograms
# plt.figure(figsize=(20, 4)) # width and height of the plot
# legends = []
# for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
#   if isinstance(layer, Tanh):
#     t = layer.out.grad
#     print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
#     hy, hx = torch.histogram(t, density=True)
#     plt.plot(hx[:-1].detach(), hy.detach())
#     legends.append(f'layer {i} ({layer.__class__.__name__}')
# plt.legend(legends);
# plt.title('gradient distribution')

# # visualize histograms
# plt.figure(figsize=(20, 4)) # width and height of the plot
# legends = []
# for i,p in enumerate(parameters):
#   t = p.grad
#   if p.ndim == 2:
#     print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
#     hy, hx = torch.histogram(t, density=True)
#     plt.plot(hx[:-1].detach(), hy.detach())
#     legends.append(f'{i} {tuple(p.shape)}')
# plt.legend(legends)
# plt.title('weights gradient distribution');


# plt.figure(figsize=(20, 4))
# legends = []
# for i,p in enumerate(parameters):
#   if p.ndim == 2:
#     plt.plot([ud[j][i] for j in range(len(ud))])
#     legends.append('param %d' % i)
# plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
# plt.legend(legends);





@torch.no_grad()
def split_loss(split):
    """ Compute the loss on the given split. """
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x] # (N, BLOCK_SIZE, N_EMBD)
    x = emb.view(emb.shape[0], -1) # concat into (N, BLOCK_SIZE * N_EMBD)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y)
    print(split, loss.item())

# put layers into eval mode
for layer in layers:
    layer.training = False
split_loss('train')
split_loss('val')

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * BLOCK_SIZE # initialize with all ...
    while True:
        # forward pass the neural net
        emb = C[torch.tensor([context])] # (1,BLOCK_SIZE,N_EMBD)
        x = emb.view(emb.shape[0], -1) # concatenate the vectors
        for layer in layers:
            x = layer(x)
        logits = x
        probs = F.softmax(logits, dim=1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token, break
        if ix == 0:
            break

    print(''.join(itos[i] for i in out)) # decode and print the generated word
