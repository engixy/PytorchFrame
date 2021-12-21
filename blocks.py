# -- coding: utf-8 --
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchsummary import summary

class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', act='relu', pad_type='zero',
                 use_bias=True, use_sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaIN2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)
        if use_sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', act='relu', use_sn=False):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class ResBlock(nn.Module):
    # 侧边分支最后一个一定不能是relu，残差块残差块，有正有负才能修复identity， 一般情况下 conv+norm+act+conv+norm
    # 据说 norm+act+conv+norm+act+conv 更好
    def __init__(self, dim, norm='in', act='relu', pad_type='zero', use_sn=False):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(Conv2dBlock(dim, dim, 3, 1, 1,
                                               norm=norm,
                                               act=act,
                                               pad_type=pad_type, use_sn=use_sn),
                                   Conv2dBlock(dim, dim, 3, 1, 1,
                                               norm=norm,
                                               act='none',
                                               pad_type=pad_type, use_sn=use_sn))

    def forward(self, x):
        x_org = x
        residual = self.model(x)
        out = x_org + 0.1 * residual    # 残差块的权重不一定要加
        return out


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, act, pad_type, use_sn=False):
        super(ResBlocks, self).__init__()
        self.model = nn.ModuleList()
        for i in range(num_blocks):
            self.model.append(ResBlock(dim, norm=norm, act=act, pad_type=pad_type, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class AdaIN2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super(AdaIN2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))  # 在内存中定义一个常量，
            self.register_buffer('running_var', torch.ones(num_features))  # 模型保存和加载的时候可以写入和读出。
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "AdaIN params are None"
        N, C, H, W = x.size()
        running_mean = self.running_mean.repeat(N)
        running_var = self.running_var.repeat(N)
        x_ = x.contiguous().view(1, N * C, H * W)   # 深拷贝+维度变换
        normed = F.batch_norm(x_, running_mean, running_var,
                              self.weight, self.bias,
                              True, self.momentum, self.eps)
        return normed.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(num_features=' + str(self.num_features) + ')'    # 输出类名加形参


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(FRN, self).__init__()
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(x**2, dim=[2, 3], keepdim=True) + self.eps)
        return torch.max(self.gamma * x + self.beta, self.tau)


class Encoder:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        # self.model = inception_v3(pretrained=True)

    def cnt(self):
        model_dict = self.model.state_dict()

        return model_dict


if __name__ == '__main__':
    E = models.resnet50(pretrained=True).cuda()
    # E.fc = nn.Linear()
    # print(E.fc.in_features)
    E.fc = nn.Linear(2,2)
    summary(E, (3, 256, 256))

    # E = Encoder()
    # cnt = E.cnt()
    # print(cnt.keys())
    # print(cnt['fc.bias'])
