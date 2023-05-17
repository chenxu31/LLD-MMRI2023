# -*- coding: utf-8 -*-
"""
Created on Mar/07/2017

@author: chenxu
"""


import math
import numpy
import platform
import pdb
import torch
import time
import os
import timm
from timm.models.layers import to_2tuple

"""
try:
    import fcntl

    LOCK_EX = fcntl.LOCK_EX
except ImportError:
    # Windows平台下没有fcntl模块
    fcntl = None
    import win32con
    import win32file
    import pywintypes

    LOCK_EX = win32con.LOCKFILE_EXCLUSIVE_LOCK
    overlapped = pywintypes.OVERLAPPED()
"""


NUM_FILTERS = [32, 64, 128, 256, 512, 1024]


class Mish(torch.nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super(Mish, self).__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input * torch.tanh(torch.nn.functional.softplus(input))


def native_act(act_type, inplace=False):
    if act_type == "prelu":
        return torch.nn.PReLU()
    elif act_type == "lrelu":
        return torch.nn.LeakyReLU(0.2, inplace=inplace)
    elif act_type == "relu":
        return torch.nn.ReLU(inplace=inplace)
    elif act_type == "gelu":
        return torch.nn.GELU()
    elif act_type == "silu":
        return torch.nn.SiLU()
    elif act_type == "mish":
        return Mish()
    elif act_type == "":
        return input
    else:
        print("Unknown act type:%s" % act_type)
        assert 0


def native_norm(norm_type, num_features, dim=2):
    if norm_type == "batch":
        if dim == 1:
            return torch.nn.BatchNorm1d(num_features)
        elif dim == 2:
            return torch.nn.BatchNorm2d(num_features)
        elif dim == 3:
            return torch.nn.BatchNorm3d(num_features)
    elif norm_type == "group":
        return torch.nn.GroupNorm(32 if num_features > 32 else num_features // 2, num_features)
    elif norm_type == "layer":
        return torch.nn.LayerNorm(num_features)
    elif norm_type == "instance":
        if dim == 1:
            return torch.nn.InstanceNorm1d(num_features)
        elif dim == 2:
            return torch.nn.InstanceNorm2d(num_features)
        elif dim == 3:
            return torch.nn.InstanceNorm3d(num_features)
    elif norm_type == "adain":
        assert dim == 2
        return AdaptiveInstanceNorm2D(num_features)
    elif norm_type == "lin":
        assert dim == 2
        return LIN(num_features)
    else:
        print("Unknown norm type:%s" % norm_type)
        assert 0


class Flatten(torch.nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()

        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return torch.flatten(input, self.start_dim, self.end_dim)


class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, dim=2, drop_path=0., norm_type="group", act_type="relu"):
        super(ResidualBlock, self).__init__()

        self.drop_path = timm.models.layers.DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()

        if dim == 1:
            conv = torch.nn.Conv1d
        elif dim == 2:
            conv = torch.nn.Conv2d
        elif dim == 3:
            conv = torch.nn.Conv3d
        else:
            assert 0

        if norm_type == "sn": # spectral norm
            ops = [torch.nn.utils.spectral_norm(conv(num_channels, num_channels, 3, 1, 1)),]
        else:
            ops = [conv(num_channels, num_channels, 3, 1, 1),]
            if norm_type:
                ops.append(native_norm(norm_type, num_channels, dim=dim))
        ops.append(native_act(act_type))

        if norm_type == "sn": # spectral norm
            ops.append(torch.nn.utils.spectral_norm(conv(num_channels, num_channels, 3, 1, 1)))
        else:
            ops.append(conv(num_channels, num_channels, 3, 1, 1))
            if norm_type:
                ops.append(native_norm(norm_type, num_channels, dim=dim))

        self.basic = torch.nn.Sequential(*ops)
        self.act = native_act(act_type) # 激活

    def forward(self, x):
        out = self.basic(x)
        return self.act(x + self.drop_path(out))


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1.
        #smooth = 0.0001

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        #loss = 1 - loss.sum() / N
        #loss = 2. * intersection.sum(1) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 0 - loss.sum() / N

        return loss


class MulticlassDiceLoss(torch.nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):
        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
        dice = DiceLoss()
        totalLoss = 0

        for i in range(1, C):
            diceLoss = dice(input[:, i], target[:, i])

            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss / (C - 1)


class SobelEdgeLoss(torch.nn.Module):
    def __init__(self, in_ch):
        super(SobelEdgeLoss, self).__init__()

        x_filter = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], numpy.float32)
        y_filter = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], numpy.float32)
        x_filter = numpy.tile(numpy.expand_dims(numpy.expand_dims(x_filter, 0), 0), (in_ch, 1, 1, 1))
        y_filter = numpy.tile(numpy.expand_dims(numpy.expand_dims(y_filter, 0), 0), (in_ch, 1, 1, 1))

        x_weight = torch.autograd.Variable(torch.from_numpy(x_filter))
        y_weight = torch.autograd.Variable(torch.from_numpy(y_filter))

        self.conv_x = torch.nn.Conv2d(in_ch, in_ch, 3, 1, groups=in_ch, bias=False)
        self.conv_y = torch.nn.Conv2d(in_ch, in_ch, 3, 1, groups=in_ch, bias=False)

        self.conv_x.weight = torch.nn.Parameter(x_weight, requires_grad=False)
        self.conv_y.weight = torch.nn.Parameter(y_weight, requires_grad=False)

        self.loss_mse = torch.nn.MSELoss()

    def forward(self, input, target):
        #return self.conv_x(input), self.conv_y(input) ####---- for test
        loss_x = self.loss_mse(self.conv_x(input), self.conv_x(target))

        loss_y = self.loss_mse(self.conv_y(input), self.conv_y(target))

        return loss_x + loss_y


class PatchNCELoss(torch.nn.Module):
    def __init__(self, nce_includes_all_negatives_from_minibatch=False, nce_T=0.07):
        super().__init__()
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.nce_T = nce_T
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool #torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        pdb.set_trace()

        # pos logit
        l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = feat_q.shape[0] #self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))

        return loss


class AdaptiveInstanceNorm2D(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = torch.nn.functional.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2D":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2D":
            num_adain_params += 2*m.num_features
    return num_adain_params


def depthwise_conv(in_ch, out_ch, dim=2, kernel_size=3, stride=1):
    if dim == 1:
        conv_ob = torch.nn.Conv1d
    elif dim == 2:
        conv_ob = torch.nn.Conv2d
    elif dim == 3:
        conv_ob = torch.nn.Conv3d
    else:
        assert 0

    kernel_size = to_2tuple(kernel_size)

    return torch.nn.Sequential(
        conv_ob(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=[x // 2 for x in kernel_size], groups=in_ch),
        conv_ob(in_ch, out_ch, kernel_size=1),
    )


def depthwise_deconv(in_ch, out_ch, dim=2, kernel_size=3, stride=2, padding=1):
    if dim == 1:
        conv_ob = torch.nn.Conv1d
        deconv_ob = torch.nn.ConvTranspose1d
    elif dim == 2:
        conv_ob = torch.nn.Conv2d
        deconv_ob = torch.nn.ConvTranspose2d
    elif dim == 3:
        conv_ob = torch.nn.Conv3d
        deconv_ob = torch.nn.ConvTranspose3d
    else:
        assert 0

    return torch.nn.Sequential(
        deconv_ob(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch,
                  output_padding=1 if padding == 1 else 0),
        conv_ob(in_ch, out_ch, kernel_size=1),
    )


class DepthwiseResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, dim=2, kernel_size=3, drop_path=0., norm_type="group", act_type="relu"):
        super(DepthwiseResidualBlock, self).__init__()

        self.drop_path = timm.models.layers.DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()

        self.basic = torch.nn.Sequential(
            depthwise_conv(num_channels, num_channels, dim=dim, kernel_size=kernel_size),
            native_norm(norm_type, num_channels, dim=dim),
            native_act(act_type),
            depthwise_conv(num_channels, num_channels, dim=dim, kernel_size=kernel_size),
            native_norm(norm_type, num_channels, dim=dim),
        )
        self.act = native_act(act_type) # 激活

    def forward(self, x):
        out = self.basic(x)
        return self.act(x + self.drop_path(out))


class PConv(torch.nn.Module):
    def __init__(self, ch, n_div, forward="split_cat", dim=2):
        super(PConv, self).__init__()
        self.ch_conv3 = ch // n_div
        self.ch_untouched = ch - self.ch_conv3
        if dim == 1:
            conv_ob = torch.nn.Conv1d
        elif dim == 2:
            conv_ob = torch.nn.Conv2d
        elif dim == 3:
            conv_ob = torch.nn.Conv3d
        else:
            assert 0

        self.partial_conv3 = conv_ob(self.ch_conv3, self.ch_conv3, kernel_size=3, stride=1, padding=1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.ch_conv3, :, :] = self.partial_conv3(x[:, :self.ch_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.ch_conv3, self.ch_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class FasterNetBlock(torch.nn.Module):

    def __init__(self,
                 ch,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0,
                 layer_scale_init_value=0,
                 norm_type="batch",
                 act_type="relu",
                 pconv_fw_type="split_cat",
                 dim=2,
                 ):
        super(FasterNetBlock, self).__init__()
        self.ch = ch
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        self.n_div = n_div

        mlp_hidden_ch = int(ch * mlp_ratio)

        if dim == 1:
            conv_ob = torch.nn.Conv1d
        elif dim == 2:
            conv_ob = torch.nn.Conv2d
        elif dim == 3:
            conv_ob = torch.nn.Conv3d
        else:
            assert 0

        mlp_layer = [
            conv_ob(ch, mlp_hidden_ch, 1, bias=False),
            native_norm(norm_type, mlp_hidden_ch, dim),
            native_act(act_type),
            conv_ob(mlp_hidden_ch, ch, 1, bias=False)
        ]

        self.mlp = torch.nn.Sequential(*mlp_layer)

        self.spatial_mixing = PConv(ch, n_div, forward=pconv_fw_type, dim=dim)

        if layer_scale_init_value > 0:
            self.layer_scale = torch.nn.Parameter(layer_scale_init_value * torch.ones((ch)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class ResnetAdaLINBlock(torch.nn.Module):
    def __init__(self, num_channels, dim=2, act_type="relu", use_bias=False):
        super(ResnetAdaLINBlock, self).__init__()

        if dim == 1:
            conv = torch.nn.Conv1d
        elif dim == 2:
            conv = torch.nn.Conv2d
        elif dim == 3:
            conv = torch.nn.Conv3d
        else:
            assert 0

        self.conv1 = conv(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.norm1 = AdaLIN(num_channels)
        self.act1 = native_act(act_type)
        self.conv2 = conv(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.norm2 = AdaLIN(num_channels)
        self.act2 = native_act(act_type)

    def forward(self, x, gamma, beta):
        out = self.conv1(x)
        out = self.norm1(out, gamma, beta)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return self.act2(out + x)


class AdaLIN(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(AdaLIN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features

        if self.using_bn:
            self.rho = torch.nn.parameter.Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(3)
            self.rho[:, :, 1].data.fill_(1)
            self.rho[:, :, 2].data.fill_(1)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = torch.nn.parameter.Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(3.2)
            self.rho[:, :, 1].data.fill_(1)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        softmax = torch.nn.Softmax(2)
        rho = softmax(self.rho)

        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3],
                                                                                            keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]

            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln

        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class ArcMarginFC(torch.nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(input),
                                            torch.nn.functional.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class ArcMarginConv(torch.nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, dim=2, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginConv, self).__init__()
        assert dim in (1, 2, 3)

        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        self.s = s
        self.m = m

        if dim == 1:
            weight_shape = (out_features, in_features, 1)
        elif dim == 2:
            weight_shape = (out_features, in_features, 1, 1)
        else:
            weight_shape = (out_features, in_features, 1, 1, 1)

        self.weight = torch.nn.Parameter(torch.FloatTensor(*weight_shape))
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def eval(self, input):
        if self.dim == 1:
            conv_op = torch.nn.functional.conv1d
        elif self.dim == 2:
            conv_op = torch.nn.functional.conv2d
        else:
            conv_op = torch.nn.functional.conv3d

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        with torch.no_grad():
            output = conv_op(torch.nn.functional.normalize(input),
                             torch.nn.functional.normalize(self.weight))

        return output

    def forward(self, input, label=None):
        if self.dim == 1:
            conv_op = torch.nn.functional.conv1d
        elif self.dim == 2:
            conv_op = torch.nn.functional.conv2d
        else:
            conv_op = torch.nn.functional.conv3d

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = conv_op(torch.nn.functional.normalize(input),
                         torch.nn.functional.normalize(self.weight))

        if label is None:
            #assert not self.training
            return cosine

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-8, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=label.device)
        one_hot.scatter_(1, label.unsqueeze(1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class LIN(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(LIN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features

        if self.using_bn:
            self.rho = torch.nn.parameter.Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3)
            self.rho[:, :, 2].data.fill_(3)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = torch.nn.parameter.Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3.2)

        self.gamma = torch.nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = torch.nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)

        softmax = torch.nn.Softmax(2)
        rho = softmax(self.rho)

        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3],
                                                                                            keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]

            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln

        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


class NetBase(torch.nn.Module):
    def __init__(self, num_downsampling=3):
        super(NetBase, self).__init__()

        self.num_downsampling = num_downsampling

    def ConvBlock(self, in_ch, out_ch, norm_type="group", act_type="relu"):
        return [
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            native_norm(norm_type, out_ch),
            native_act(act_type),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            native_norm(norm_type, out_ch),
            native_act(act_type),
        ]

    def build_encoder(self, in_ch, num_rb=0, norm_type="group", act_type="relu"):
        net_list = []
        for i in range(self.num_downsampling):
            net_list.append(torch.nn.Conv2d(in_ch if i == 0 else NUM_FILTERS[i], NUM_FILTERS[i], 3, padding=1))
            net_list.append(native_norm(norm_type, NUM_FILTERS[i]))
            net_list.append(native_act(act_type))

            net_list.append(torch.nn.Conv2d(NUM_FILTERS[i], NUM_FILTERS[i], 3, padding=1))
            net_list.append(native_norm(norm_type, NUM_FILTERS[i]))
            net_list.append(native_act(act_type))

            net_list.append(torch.nn.Conv2d(NUM_FILTERS[i], NUM_FILTERS[i + 1], 3, stride=2, padding=1))
            net_list.append(native_norm(norm_type, NUM_FILTERS[i + 1]))
            net_list.append(native_act(act_type))

        for i in range(num_rb):
            net_list.append(ResidualBlock(NUM_FILTERS[self.num_downsampling], norm_type=norm_type, act_type=act_type))

        return torch.nn.Sequential(*net_list)

    def build_decoder(self, in_ch, out_ch, num_rb=0, norm_type="group", act_type="relu"):
        net_list = []
        for i in range(num_rb):
            net_list.append(ResidualBlock(in_ch, norm_type=norm_type, act_type=act_type))

        for i in range(self.num_downsampling):
            ch = NUM_FILTERS[self.num_downsampling - i - 1]

            net_list.append(torch.nn.ConvTranspose2d(NUM_FILTERS[self.num_downsampling - i], ch, 3, stride=2, padding=1, output_padding=1))
            net_list.append(native_norm(norm_type, ch))
            net_list.append(native_act(act_type))

            net_list.append(torch.nn.Conv2d(ch, ch, 3, padding=1))
            net_list.append(native_norm(norm_type, ch))
            net_list.append(native_act(act_type))

            net_list.append(torch.nn.Conv2d(ch, ch, 3, padding=1))
            net_list.append(native_norm(norm_type, ch))
            net_list.append(native_act(act_type))

        net_list.append(torch.nn.Conv2d(NUM_FILTERS[0], out_ch, 1))

        return torch.nn.Sequential(*net_list)

    def forward(self, *params):
        pass


class NetUNetBaseNew(NetBase):
    def __init__(self, in_ch, out_ch=0, inner_ch=32, dim=2, depths=(1, 1, 2, 4), drop_path_rate=0.1,
                 down_type="conv", up_type="conv", norm_type="group", act_type="relu"):
        super(NetUNetBaseNew, self).__init__(num_downsampling=len(depths) - 1)

        assert down_type in ("conv", "pool")
        assert up_type in ("conv", "inter")

        self.up_type = up_type
        self.dim = dim
        self.depths = depths

        num_filters = [inner_ch * (2 ** i) for i in range(len(depths))]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if dim == 1:
            conv_ob = torch.nn.Conv1d
            pool_ob = torch.nn.MaxPool1d
        elif dim == 2:
            conv_ob = torch.nn.Conv2d
            pool_ob = torch.nn.MaxPool2d
        elif dim == 3:
            conv_ob = torch.nn.Conv3d
            pool_ob = torch.nn.MaxPool3d
        else:
            assert 0

        self.input_conv = torch.nn.Sequential(
            conv_ob(in_ch, num_filters[0], 3, padding=1),
            native_norm(norm_type, num_filters[0], dim=dim),
            native_act(act_type),
        )

        for i in range(len(depths) - 1):
            stage_dpr = dpr[sum(depths[:i]):sum(depths[:i + 1])]

            if down_type == "conv":
                setattr(self, "encoder_down_%d" % i, torch.nn.Sequential(
                    depthwise_conv(num_filters[i], num_filters[i + 1], dim=dim, kernel_size=3, stride=2),
                    native_norm(norm_type, num_filters[i + 1], dim=dim),
                    native_act(act_type),
                ))
            else:
                setattr(self, "encoder_down_%d" % i, torch.nn.Sequential(
                    pool_ob(2),
                    conv_ob(num_filters[i], num_filters[i + 1], kernel_size=1),
                    native_norm(norm_type, num_filters[i + 1], dim=dim),
                    native_act(act_type),
                ))

            encoder_blocks = []
            for j in range(depths[i]):
                encoder_blocks.append(DepthwiseResidualBlock(num_filters[i], dim=dim, drop_path=stage_dpr[j],
                                                             norm_type=norm_type, act_type=act_type))
            setattr(self, "encoder_block_%d" % i, torch.nn.Sequential(*encoder_blocks))

            if up_type == "conv":
                setattr(self, "decoder_up_%d" % i, torch.nn.Sequential(
                    depthwise_deconv(num_filters[len(depths) - i - 1], num_filters[len(depths) - i - 2], dim=dim),
                    native_norm(norm_type, num_filters[len(depths) - i - 2]),
                    native_act(act_type),
                ))
            else:
                setattr(self, "decoder_up_%d" % i, torch.nn.Sequential(
                    conv_ob(num_filters[len(depths) - i - 1], num_filters[len(depths) - i - 2], kernel_size=1),
                    native_norm(norm_type, num_filters[len(depths) - i - 2]),
                    native_act(act_type),
                ))

            decoder_blocks = [
                conv_ob(num_filters[len(depths) - i - 2] * 2, num_filters[len(depths) - i - 2], kernel_size=1),
                native_norm(norm_type, num_filters[len(depths) - i - 2], dim=dim),
                native_act(act_type),
            ]
            stage_decoder_dpr = dpr[sum(depths[:len(depths) - i - 2]):sum(depths[:len(depths) - i - 1])][::-1]
            for j in range(depths[-(i + 2)]):
                decoder_blocks.append(DepthwiseResidualBlock(num_filters[len(depths) - i - 2], dim=dim,
                                                             drop_path=stage_decoder_dpr[j], norm_type=norm_type, act_type=act_type))

            setattr(self, "decoder_block_%d" % i, torch.nn.Sequential(*decoder_blocks))

        bridge = []
        stage_dpr = dpr[sum(depths[:len(depths) - 1]):sum(depths[:len(depths)])]
        for j in range(depths[-1]):
            bridge.append(DepthwiseResidualBlock(num_filters[-1], dim=dim, drop_path=stage_dpr[j],
                                                 norm_type=norm_type, act_type=act_type))
        self.bridge = torch.nn.Sequential(*bridge)

        if out_ch > 0:
            self.decoder_output = conv_ob(num_filters[0], out_ch, 3, padding=1)
        else:
            self.decoder_output = None

    def forward(self, im):
        x = self.input_conv(im)

        features = []
        for i in range(len(self.depths) - 1):
            x = getattr(self, "encoder_block_%d" % i)(x)
            features.append(x)
            x = getattr(self, "encoder_down_%d" % i)(x)

        x = self.bridge(x)

        for i in range(len(self.depths) - 1):
            if self.up_type == "conv":
                x = getattr(self, "decoder_up_%d" % i)(x)
            else:
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode='trilinear' if self.dim == 3 else 'bilinear',
                                                    align_corners=True)
                x = getattr(self, "decoder_up_%d" % i)(x)
            x = torch.cat([x, features[-(i + 1)]], dim=1)
            x = getattr(self, "decoder_block_%d" % i)(x)

        output = x

        if self.decoder_output:
            output = self.decoder_output(output)

        return output


class NetUNetBase(NetBase):
    def __init__(self, num_downsampling, in_ch, out_ch, inner_ch=32, need_output_conv=True, down_type="conv",
                 up_type="conv", norm_type="group", act_type="relu"):
        super(NetUNetBase, self).__init__(num_downsampling)

        assert down_type in ("conv", "pool")
        assert up_type in ("conv", "inter")

        self.up_type = up_type
        self.need_output_conv = need_output_conv

        num_filters = [inner_ch * (2 ** i) for i in range(num_downsampling + 1)]

        for i in range(self.num_downsampling):
            if down_type == "conv":
                setattr(self, "encoder_down_%d" % i, torch.nn.Sequential(
                    torch.nn.Conv2d(num_filters[i], num_filters[i + 1], 3, stride=2, padding=1),
                    native_norm(norm_type, num_filters[i + 1]),
                    native_act(act_type),
                ))
                ch = in_ch if i == 0 else num_filters[i]
            else:
                setattr(self, "encoder_down_%d" % i, torch.nn.MaxPool2d(2))
                ch = in_ch if i == 0 else num_filters[i - 1]

            setattr(self, "encoder_block_%d" % i, torch.nn.Sequential(
                *self.ConvBlock(ch, num_filters[i], norm_type=norm_type, act_type=act_type)))

            if up_type == "conv":
                setattr(self, "decoder_up_%d" % i, torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(num_filters[self.num_downsampling - i], num_filters[self.num_downsampling - i - 1],
                                             3, stride=2, padding=1, output_padding=1),
                    native_norm(norm_type, num_filters[self.num_downsampling - i - 1]),
                    native_act(act_type),
                ))
                ch = num_filters[self.num_downsampling - i - 1] * 2
            else:
                ch = num_filters[self.num_downsampling - i - 1] + num_filters[self.num_downsampling - i]

            setattr(self, "decoder_block_%d" % i, torch.nn.Sequential(
                *self.ConvBlock(ch, num_filters[self.num_downsampling - i - 1], norm_type=norm_type, act_type=act_type)))

        bridge = self.ConvBlock(num_filters[self.num_downsampling], num_filters[self.num_downsampling],
                                norm_type=norm_type, act_type=act_type)
        if down_type == "pool":
            bridge = [
                torch.nn.Conv2d(num_filters[self.num_downsampling - 1], num_filters[self.num_downsampling], 3, padding=1),
                native_norm(norm_type, num_filters[self.num_downsampling]),
                native_act(act_type),
            ] + bridge
        self.bridge = torch.nn.Sequential(*bridge)

        if self.need_output_conv:
            self.decoder_output = torch.nn.Conv2d(num_filters[0], out_ch, 3, padding=1)

    def forward(self, im):
        x = im
        features = []
        for i in range(self.num_downsampling):
            x = getattr(self, "encoder_block_%d" % i)(x)
            features.append(x)
            x = getattr(self, "encoder_down_%d" % i)(x)

        x = self.bridge(x)

        for i in range(self.num_downsampling):
            if self.up_type == "conv":
                x = getattr(self, "decoder_up_%d" % i)(x)
            else:
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, features[-(i + 1)]], dim=1)
            x = getattr(self, "decoder_block_%d" % i)(x)

        output = x

        if self.need_output_conv:
            output = self.decoder_output(output)

        return output


class ModelBase(object):
    def __init__(self, device, logger):
        self.logger = logger
        self.device = device

        self.loss_op_l1 = torch.nn.L1Loss()
        self.loss_op_mse = torch.nn.MSELoss()
        self.loss_op_ce = torch.nn.CrossEntropyLoss()
        self.loss_op_dice = MulticlassDiceLoss()
        self.loss_op_triplet = torch.nn.TripletMarginLoss(margin=10.)
        self.loss_op_l1.to(self.device)
        self.loss_op_mse.to(self.device)
        self.loss_op_ce.to(self.device)
        self.loss_op_dice.to(self.device)
        self.loss_op_triplet.to(self.device)

    def save_model(self, model, checkpoint_dir, tag="Final", enforce=False):
        while True:
            try:
                torch.save(model, os.path.join(checkpoint_dir, "%s.pth" % tag))
            except:
                self.logger.info("Save failed, retry later.")

                if not enforce:
                    break

                # try again later
                time.sleep(600)
                continue

            break

    def loss_l1(self, data, label, summary, name):
        return self.do_loss(self.loss_op_l1, data, label, summary, name)

    def loss_mse(self, data, label, summary, name):
        return self.do_loss(self.loss_op_mse, data, label, summary, name)

    def loss_ce(self, data, label, summary, name):
        return self.do_loss(self.loss_op_ce, data, label, summary, name)

    def loss_dice(self, data, label, summary, name):
        data = torch.nn.functional.softmax(data, dim=1)

        return self.do_loss(self.loss_op_dice, data,
                            torch.nn.functional.one_hot(label, NUM_CLASSES).permute((0, 4, 1, 2, 3)), summary, name)

    def loss_triplet(self, anchor, positive, negtive, summary, name):
        loss = self.loss_op_triplet(anchor, positive, negtive)

        if name not in summary:
            summary[name] = loss.detach()
        else:
            summary[name] += loss.detach()

        return loss

    def do_loss(self, loss_op, data, label, summary, name):
        if isinstance(data, list) or isinstance(data, tuple):
            assert isinstance(label, list) or isinstance(label, tuple)
            assert len(data) == len(label)
            for i in range(len(data)):
                if i == 0:
                    loss = loss_op(data[i], label[i])
                else:
                    loss += loss_op(data[i], label[i])

            loss /= len(data)
        else:
            loss = loss_op(data, label)

        if name not in summary:
            summary[name] = loss.detach()
        else:
            summary[name] += loss.detach()

        return loss


def get_patch_ids(patch_shape, data_shape, strides):
    patch_depth, patch_height, patch_width = patch_shape
    data_depth, data_height, data_width = data_shape

    depth_stride = min(patch_depth, strides[0])
    height_stride = min(patch_height, strides[1])
    width_stride = min(patch_width, strides[2])
    height_ids = list(range(0, data_height - patch_height + 1, height_stride))
    width_ids = list(range(0, data_width - patch_width + 1, width_stride))
    depth_ids = list(range(0, data_depth - patch_depth + 1, depth_stride))
    if height_ids[-1] + patch_height < data_height:
        height_ids.append(data_height - patch_height)
    if width_ids[-1] + patch_width < data_width:
        width_ids.append(data_width - patch_width)
    if depth_ids[-1] + patch_depth < data_depth:
        depth_ids.append(data_depth - patch_depth)

    return depth_ids, height_ids, width_ids


def produce_results(device, models, input_shapes, input_datas, data_shape, patch_shape, is_seg=False, num_classes=2, batch_size=16):
    if not isinstance(models, tuple) and not isinstance(models, list):
        models = [models, ]
        input_shapes = [input_shapes, ]
        input_datas = [input_datas, ]

    assert len(models) == len(input_shapes)
    assert len(models) == len(input_datas)

    patch_depth, patch_height, patch_width = patch_shape

    data_fake = []
    used = []
    batch_input = []
    for arr in input_shapes:
        local_inputs = []
        for input_shape in arr:
            local_inputs.append(numpy.zeros([batch_size] + list(input_shape), numpy.float32))
        batch_input.append(local_inputs)
    for i in range(len(models)):
        if is_seg:
            data_fake.append(numpy.zeros([num_classes] + list(data_shape), numpy.float32))
        else:
            data_fake.append(numpy.zeros(data_shape, numpy.float32))
        used.append(numpy.zeros(data_shape, numpy.float32))

    idx = 0
    batch_locs = [None] * batch_size
    for i in range(data_shape[0] - patch_depth + 1):
        batch_locs[idx] = i
        for i1 in range(len(input_datas)):
            for i2, data in enumerate(input_datas[i1]):
                if (isinstance(data, list) or isinstance(data, tuple)) and data[0] == "constant":
                    batch_input[i1][i2][idx] = data[1]
                else:
                    batch_input[i1][i2][idx] = data[i: i + patch_depth, :, :]

        idx += 1
        if idx < batch_size:
            continue

        idx = 0
        for ii, model in enumerate(models):
            ret = model(*[torch.tensor(x, device=device) for x in batch_input[ii]])

            if isinstance(ret, list) or isinstance(ret, tuple):
                ret = ret[0]

            ret = ret.cpu().detach().numpy()

            for batch_id, loc in enumerate(batch_locs):
                if is_seg:
                    data_fake[ii][:, loc:loc + patch_depth, :, :] += ret[batch_id]
                else:
                    data_fake[ii][loc:loc + patch_depth, :, :] += ret[batch_id]

                used[ii][loc:loc + patch_depth, :, :] += 1.

    if idx != 0:
        for ii, model in enumerate(models):
            ret = model(*[torch.tensor(x, device=device) for x in batch_input[ii]])

            if isinstance(ret, list) or isinstance(ret, tuple):
                ret = ret[0]

            ret = ret.cpu().numpy()

            for batch_id, loc in enumerate(batch_locs[:idx]):
                if is_seg:
                    data_fake[ii][:, loc:loc + patch_depth, :, :] += ret[batch_id]
                else:
                    data_fake[ii][loc:loc + patch_depth, :, :] += ret[batch_id]

                used[ii][loc:loc + patch_depth, :, :] += 1.

    for _used in used:
        assert _used.min() > 0

    for ii in range(len(data_fake)):
        if is_seg:
            data_fake[ii] /= numpy.expand_dims(used[ii], 0)
        else:
            data_fake[ii] /= used[ii]

    return data_fake[0] if len(data_fake) == 1 else data_fake


def produce_results_3D(device, models, input_shapes, input_datas, data_shape, patch_shape, is_seg=False,
                       num_classes=2, batch_size=16, strides=(32, 32, 32)):
    if not isinstance(models, tuple) and not isinstance(models, list):
        models = [models, ]
        input_shapes = [input_shapes, ]
        input_datas = [input_datas, ]

    assert len(models) == len(input_shapes)
    assert len(models) == len(input_datas)

    patch_depth, patch_height, patch_width = patch_shape

    depth_ids, height_ids, width_ids = get_patch_ids(patch_shape, data_shape, strides)

    data_fake = []
    used = []
    batch_input = []
    for arr in input_shapes:
        local_inputs = []
        for input_shape in arr:
            local_inputs.append(numpy.zeros([batch_size,] + list(input_shape), numpy.float32))
        batch_input.append(local_inputs)
    for i in range(len(models)):
        if is_seg:
            data_fake.append(numpy.zeros([num_classes] + list(data_shape), numpy.float32))
        else:
            data_fake.append(numpy.zeros(data_shape, numpy.float32))
        used.append(numpy.zeros(data_shape, numpy.float32))

    idx = 0
    batch_locs = [None] * batch_size
    for id_d in depth_ids:
        for id_h in height_ids:
            for id_w in width_ids:
                batch_locs[idx] = (id_d, id_h, id_w)
                for i1 in range(len(input_datas)):
                    for i2, data in enumerate(input_datas[i1]):
                        if (isinstance(data, list) or isinstance(data, tuple)) and data[0] == "constant":
                            batch_input[i1][i2][idx] = data[1]
                        else:
                            batch_input[i1][i2][idx] = data[id_d: id_d + patch_depth, id_h:id_h + patch_height, id_w:id_w + patch_width]

                idx += 1
                if idx < batch_size:
                    continue

                idx = 0
                for ii, model in enumerate(models):
                    ret = model(*[torch.tensor(x, device=device) for x in batch_input[ii]])

                    if isinstance(ret, list) or isinstance(ret, tuple):
                        ret = ret[0]

                    ret = ret.cpu().detach().numpy()

                    for batch_id, loc in enumerate(batch_locs):
                        if is_seg:
                            data_fake[ii][:, loc[0]:loc[0] + patch_depth, loc[1]:loc[1] + patch_height, loc[2]:loc[2] + patch_width] += ret[batch_id]
                        else:
                            data_fake[ii][loc[0]:loc[0] + patch_depth, loc[1]:loc[1] + patch_height, loc[2]:loc[2] + patch_width] += ret[batch_id]

                        used[ii][loc[0]:loc[0] + patch_depth, loc[1]:loc[1] + patch_height, loc[2]:loc[2] + patch_width] += 1.

    if idx != 0:
        for ii, model in enumerate(models):
            ret = model(*[torch.tensor(x, device=device) for x in batch_input[ii]])

            if isinstance(ret, list) or isinstance(ret, tuple):
                ret = ret[0]

            ret = ret.cpu().numpy()

            for batch_id, loc in enumerate(batch_locs[:idx]):
                if is_seg:
                    data_fake[ii][:, loc[0]:loc[0] + patch_depth, loc[1]:loc[1] + patch_height, loc[2]:loc[2] + patch_width] += ret[batch_id]
                else:
                    data_fake[ii][loc[0]:loc[0] + patch_depth, loc[1]:loc[1] + patch_height, loc[2]:loc[2] + patch_width] += ret[batch_id]

                used[ii][loc[0]:loc[0] + patch_depth, loc[1]:loc[1] + patch_height, loc[2]:loc[2] + patch_width] += 1.

    for _used in used:
        assert _used.min() > 0

    for ii in range(len(data_fake)):
        if is_seg:
            data_fake[ii] /= numpy.expand_dims(used[ii], 0)
        else:
            data_fake[ii] /= used[ii]

    return data_fake[0] if len(data_fake) == 1 else data_fake


"""
class Lock(object):
    # 进程锁

    def __init__(self, filename='processlock.txt'):
        self.filename = filename
        # 如果文件不存在则创建
        self.handle = open(filename, 'w')

    def acquire(self):
        # 给文件上锁
        if fcntl:
            fcntl.flock(self.handle, LOCK_EX)
        else:
            hfile = win32file._get_osfhandle(self.handle.fileno())
            win32file.LockFileEx(hfile, LOCK_EX, 0, -0x10000, overlapped)

    def release(self):
        # 文件解锁
        if fcntl:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        else:
            hfile = win32file._get_osfhandle(self.handle.fileno())
            win32file.UnlockFileEx(hfile, 0, -0x10000, overlapped)

    def __del__(self):
        try:
            self.handle.close()
            os.remove(self.filename)
        except:
            pass
"""

