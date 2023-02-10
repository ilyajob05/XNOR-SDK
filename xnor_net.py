import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, PILToTensor, Lambda

from tqdm.notebook import tqdm


class BinaryLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, alpha, betta, gamma):
        """ X : (batch_size, in_features)
            W : (in_features, out_features)
            alpha : (out_features,)
        """
        X_bin = torch.sign(X)
        W_bin = torch.sign(W)
        XW_bin = X_bin.mm(W_bin)

        assert (torch.all(alpha >= 0))
        assert (torch.all(betta >= 0))
        assert (torch.all(gamma >= 0))

        ctx.save_for_backward(X_bin, W_bin, X, W, alpha, betta, gamma)
        # k = (alpha * betta[None, :].T.mm(gamma[None, :])).flatten()
        # return XW_bin * k
        return XW_bin * (alpha * betta[None, :].T.mm(gamma[None, :])).flatten()

    @staticmethod
    def backward(ctx, dL_dY):
        """ dL_dZ : (batch_size, out_featuers)
            dL_dX : (batch_size, in_features)
            dL_dW : (in_features, out_features)
            dL_dalpha : (out_features,)
        """
        X_bin, W_bin, X, W, alpha, betta, gamma = ctx.saved_tensors

        dL_dX = None
        dL_dW = None
        dL_dalpha = None
        dL_dbetta = None
        dL_dgamma = None

        if ctx.needs_input_grad[0]:
            dL_dX = dL_dY.mm(W_bin) * (alpha * betta[None, :].T.mm(gamma[None, :])).flatten() * (1 - torch.tanh(X).pow(2))

        if ctx.needs_input_grad[1]:
            dL_dW = X_bin.T.mm(dL_dY) * (alpha * betta[None, :].T.mm(gamma[None, :])).flatten() * (1 - torch.tanh(W).pow(2))

        if ctx.needs_input_grad[2]:
            dL_dalpha = (X_bin.mm(W_bin) * (betta[None, :].T.mm(gamma[None, :])).flatten() * dL_dY).sum()

        # todo:
        if ctx.needs_input_grad[3]:
            dL_dbetta = (X_bin.mm(W_bin) * (alpha * gamma[None, :]).flatten() * dL_dY).sum(axis=0)

        if ctx.needs_input_grad[4]:
            dL_dgamma = (X_bin.mm(W_bin) * (alpha * betta[None, :].T).flatten() * dL_dY).sum(axis=0)

        return dL_dX, dL_dW, dL_dalpha, dL_dbetta, dL_dgamma


class BinaryLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.alpha = nn.Parameter(torch.empty(0))
        self.betta = nn.Parameter(torch.empty(32))
        self.gamma = nn.Parameter(torch.empty(64))

        nn.init.xavier_uniform_(self.W)
        self.alpha.data = self.W.abs().mean()
        self.betta.data = self.W.abs().mean(axis=0)[:32]
        self.gamma.data = self.W.abs().mean(axis=0)[32:96]

        self.func = BinaryLinearFunc.apply

    def forward(self, X):
        self.alpha.data.relu_()  # alpha should be > 0
        self.betta.data.relu_()  # alpha should be > 0
        self.gamma.data.relu_()  # alpha should be > 0

        return self.func(X, self.W, self.alpha, self.betta, self.gamma)


def create_net(n_hidden=4, input_dim=784, out_dim=10, hid_dim=128, binary=False):
    net = nn.Sequential()

    net.add_module("input", nn.Linear(input_dim, hid_dim, bias=False))

    if not binary:
        net.add_module("input_relu", nn.ReLU())

    for i in range(n_hidden):
        if not binary:
            net.add_module(f"hidden_{i}", nn.Linear(hid_dim, hid_dim, bias=False))
            net.add_module(f"relu_{i}", nn.ReLU())
        else:
            net.add_module(f"hidden_{i}", BinaryLinearLayer(hid_dim, hid_dim))

    net.add_module("output", nn.Linear(hid_dim, out_dim, bias=False))

    return net

tfm = Compose([
    PILToTensor(),
    Lambda(lambda x: x.flatten() / 255),
])

# train_data = MNIST("mnist", train=True, download=True, transform=tfm)
# test_data = MNIST("mnist", train=False, download=True, transform=tfm)

train_data = CIFAR10("mnist", train=True, download=True, transform=tfm)
test_data = CIFAR10("mnist", train=False, download=True, transform=tfm)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

input_dim = train_data[0][0].shape[0]


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')


def train_epoch(net, optim, loss_fn, epoch, step, log_name):
    for image, label in train_loader:
        image = image
        label = label

        output = net(image)
        loss = loss_fn(output, label)

        writer.add_scalars(f"Loss/train", {log_name: loss.item()}, step[0])

        loss.backward()
        optim.step()
        optim.zero_grad()
        step[0] += 1
        writer.flush()


@torch.no_grad()
def tst_epoch(net, loss_fn, epoch, step, log_name):
    total_loss = 0
    total_steps = 0

    total_acc = 0
    total_samples = 0

    for image, label in test_loader:
        image = image
        label = label

        output = net(image)
        loss = loss_fn(output, label)
        total_loss += loss.item()
        total_steps += 1

        total_acc += (output.argmax(dim=1) == label).sum().item()
        total_samples += image.shape[0]

    writer.add_scalars(f"Loss/test", {log_name: total_loss / total_steps}, step[0])
    writer.add_scalars(f"Accuracy/test", {log_name: total_acc / total_samples}, step[0])
    writer.flush()


simple_net = create_net(input_dim=input_dim)

optim = torch.optim.Adam(simple_net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

step_train = [0]
step_test = [0]
n_epochs = 1
max_step = n_epochs * len(train_loader)

for epoch in range(n_epochs):
    train_epoch(simple_net, optim, loss_fn, epoch, step_train, 'float')
    tst_epoch(simple_net, loss_fn, epoch, step_test, 'float')


bin_net = create_net(input_dim=input_dim, hid_dim=2048, binary=True)

optim = torch.optim.Adam(bin_net.parameters(), lr=1e-4)

step_train = [0]
step_test = [0]
n_epochs = 1
max_step = n_epochs * len(train_loader)

min_t = 1
max_t = 2

for epoch in range(n_epochs):
    train_epoch(bin_net, optim, loss_fn, epoch, step_train, 'bin')
    tst_epoch(bin_net, loss_fn, epoch, step_test, 'bin')

print('end')
