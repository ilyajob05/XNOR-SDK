import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple

from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, PILToTensor, ToTensor, Lambda, transforms

from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t1 = torch.ones(3, 3, device = device)
print(t1)


class BinaryActivationFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        result = torch.sign(input)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.le(-1)] = 0
        grad_input[input.ge(-1)] = 0
        return grad_input


class BinaryConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = (1,1), padding = (0,0), dilation = (1,1), groups = 1,
                 bias = True, padding_mode: str = 'zeros', device = None, dtype = None, out_width = 1, out_height = 1) -> None:
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias, padding_mode, device, dtype)
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        if bias:
            self.bias = torch.autograd.Parameter(torch.empty(out_channels, **(self.factory_kwargs)))
        else:
            self.register_parameter('bias', None)
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(out_height).reshape(1,-1,1), requires_grad=False)
        self.beta = nn.Parameter(torch.ones(out_width).reshape(1,1,-1), requires_grad=False)
        self.gamma = nn.Parameter(torch.ones(out_channels).reshape(-1,1,1), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = BinaryActivationFunc.apply(input)
        real_weight = self.weight
        mean_weights = real_weight.mul(-1).mean(dim=1, keepdim=True).expand_as(self.weight).contiguous()
        centered_weights = real_weight.add(mean_weights)
        cliped_weights = torch.clamp(centered_weights, -1.0, 1.0)
        signed_weights = torch.sign(centered_weights).detach() - cliped_weights.detach() + cliped_weights
        binary_weights = signed_weights
        input = torch.nn.functional.conv2d(input, binary_weights, bias=self.bias, stride=self.stride, padding=self.padding,
                                           dilation=self.dilation, groups=self.groups)
        return input.mul(self.gamma).mul(self.beta).mul(self.alpha)



# def BinaryConvolution(
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride = 1,
#         padding = 0,
#         dilation = 1,
#         groups = 1,
#         bias = True,
#         padding_mode: str = 'zeros',  # TODO: refine this type
#         device=None,
#         dtype=None
# ):
#     sq = torch.nn.Sequential()
#     sq.add_module('input', BinaryActivationFunc(nn.BatchNorm2d(in_channels, eps=1.0e-4)))
#     # sq.add_module('bin_activation', BinaryActivationFunc())
#     sq.add_module('conv', nn.Conv2d(in_channels, out_channels,kernel_size,stride,padding,dilation, groups, bias, padding_mode, device, dtype))
#     return sq


def BinaryMaxConvolution(
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
):
    sq = torch.nn.Sequential()
    sq.add_module('input', nn.BatchNorm2d(in_channels, eps=1.0e-4))
    sq.add_module('bin_activation', BinaryActivationFunc())
    sq.add_module('conv', nn.Conv2d(in_channels, out_channels,kernel_size,stride,padding,dilation, groups, bias, padding_mode, device, dtype))
    sq.add_module('max_pool', nn.MaxPool2d(3,2))
    return sq


class NetBin(nn.Module):
    def __init__(self):
        super(NetBin, self).__init__()
        # convolution 1
        self.bn0 = nn.BatchNorm2d(3)
        self.conv00 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu00 = nn.PReLU()
        self.conv0 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu0 = nn.PReLU()
        self.pool0 = nn.MaxPool2d(2)
        self.drop0 = nn.Dropout2d()

        # convolution 2 bin
        self.bn1 = nn.BatchNorm2d(32)
        self.conv10 = BinaryConv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv1 = BinaryConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d()

        # convolution 3
        self.bn2 = nn.BatchNorm2d(64)
        self.conv20 = BinaryConv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu20 = nn.PReLU()
        self.conv2 = BinaryConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d()

        # full connection layer 1
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 10)
        self.fc1Prelu = nn.PReLU()

        # full connection layer 2
        self.fc2 = nn.Linear(10, 10)

    # direct computation
    def forward(self, x):
        x = self.bn0(x)
        x = self.conv00(x)
        x = self.relu00(x)
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.drop0(x)

        x = self.bn1(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.bn2(x)
        x = self.conv20(x)
        x = self.relu20(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        #         show input parameter
        #         print(x.shape)
        x = self.bn3(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc1Prelu(x)
        x = torch.nn.functional.dropout(x)
        x = self.fc2(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


# network initialization
net_bin = NetBin().to(device)
# show network
print(net_bin)










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
            dL_dalpha = (X_bin.mm(W_bin) * (betta[None, :].T.mm(gamma[None, :])).flatten() * dL_dY).sum(axis=0)

        # todo:
        if ctx.needs_input_grad[3]:
            dL_dbetta = (X_bin.mm(W_bin) * (alpha * gamma.diag().flatten()[None, :] * dL_dY)).reshape(128,32,-1).sum(axis=-1)

        if ctx.needs_input_grad[4]:
            dL_dgamma = (X_bin.mm(W_bin) * (alpha * betta.diag().flatten()[None, :] * dL_dY)).reshape(128,32,-1).sum(axis=-1)

        return dL_dX, dL_dW, dL_dalpha, dL_dbetta, dL_dgamma


class BinaryLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.alpha = nn.Parameter(torch.empty(0))
        self.betta = nn.Parameter(torch.empty(32))
        self.gamma = nn.Parameter(torch.empty(32))

        nn.init.xavier_uniform_(self.W)
        self.alpha.data = self.W.abs().mean()
        self.betta.data = self.W.abs().mean(axis=0)[:32]
        self.gamma.data = self.W.abs().mean(axis=0)[32:64]

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

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
tfm = Compose([ToTensor(), normalize, ])

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
        image = image.to(device)
        label = label.to(device)

        output = net(image)
        loss = loss_fn(output, label)

        writer.add_scalars(f"Loss/train", {log_name: loss.item()}, step[0])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0e-4)
        optim.step()
        optim.zero_grad()
        step[0] += 1
        writer.flush()

        writer.add_histogram('fc2', net.fc2.weight, epoch)

        writer.add_histogram('alpha_conv1', net.conv1.alpha.data, epoch)

    writer.add_hparams({'alpha_conv1': net.conv1.alpha.data.item(),
                        'betta_conv1': net.conv1.betta.data.sum().item(),
                        'alpha_conv10': net.conv10.alpha.data.item(),
                        'betta_conv10': net.conv10.betta.data.sum().item(),
                        'betta_conv2': net.conv2.betta.data.sum().item(),
                        'alpha_conv2': net.conv2.alpha.data.item(),
                        'betta_conv20': net.conv20.betta.data.sum().item(),
                        'alpha_conv20': net.conv20.alpha.data.item(),
                        },
                       {f"Loss/train{log_name}": loss.item()})

    writer.add_histogram('conv00', net.conv00.weight, epoch)
    writer.add_histogram('conv0', net.conv0.weight, epoch)
    writer.add_histogram('conv1', net.conv1.weight, epoch)
    writer.add_histogram('conv10', net.conv10.weight, epoch)
    writer.add_histogram('conv2', net.conv2.weight, epoch)
    writer.add_histogram('conv20', net.conv20.weight, epoch)
    writer.add_histogram('fc1', net.fc1.weight, epoch)
    writer.add_histogram('fc2', net.fc2.weight, epoch)

    writer.add_histogram('alpha_conv1', net.conv1.alpha.data, epoch)
    writer.add_histogram('betta_conv1', net.conv1.betta.data, epoch)
    writer.add_histogram('alpha_conv10', net.conv10.alpha.data, epoch)
    writer.add_histogram('betta_conv10', net.conv10.betta.data, epoch)
    writer.add_histogram('betta_conv2', net.conv2.betta.data, epoch)
    writer.add_histogram('alpha_conv2', net.conv2.alpha.data, epoch)
    writer.add_histogram('betta_conv20', net.conv20.betta.data, epoch)
    writer.add_histogram('alpha_conv20', net.conv20.alpha.data, epoch)
    writer.add_histogram('fc2', net.fc2.weight, epoch)


@torch.no_grad()
def tst_epoch(net, loss_fn, epoch, step, log_name):
    total_loss = 0
    total_steps = 0

    total_acc = 0
    total_samples = 0

    for image, label in test_loader:
        image = image.to(device)
        label = label.to(device)

        output = net(image)
        loss = loss_fn(output, label)
        total_loss += loss.item()
        total_steps += 1
        total_acc += (output.argmax(dim=1) == label).sum().item()
        total_samples += image.shape[0]

    writer.add_scalars(f"Loss/test", {log_name: total_loss / total_steps}, step[0])
    writer.add_scalars(f"Accuracy/test", {log_name: total_acc / total_samples}, step[0])
    step[0] += 1
    writer.flush()



bin_net = create_net(input_dim=input_dim, hid_dim=1024, binary=True).to(device)

optim_l = torch.optim.RAdam(net_bin.parameters(), lr=1.0e-5)
optim = torch.optim.RAdam(net_bin.parameters(), lr=5.0e-5)
# optim = torch.optim.ASGD(bin_net.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

step_train = [0]
step_test = [0]
n_epochs = 100
max_step = n_epochs * len(train_loader)

min_t = 1
max_t = 2

for epoch in range(n_epochs):
    if epoch == 0:
        train_epoch(net_bin, optim_l, loss_fn, epoch, step_train, 'bin')
        tst_epoch(net_bin, loss_fn, epoch, step_test, 'bin')

    train_epoch(net_bin, optim, loss_fn, epoch, step_train, 'bin')
    tst_epoch(net_bin, loss_fn, epoch, step_test, 'bin')



simple_net = create_net(input_dim=input_dim).to(device)

optim = torch.optim.Adam(simple_net.parameters(), lr=1e-3)
step_train = [0]
step_test = [0]
n_epochs = 10
max_step = n_epochs * len(train_loader)

for epoch in range(n_epochs):
    train_epoch(simple_net, optim, loss_fn, epoch, step_train, 'float')
    tst_epoch(simple_net, loss_fn, epoch, step_test, 'float')





print('end')

