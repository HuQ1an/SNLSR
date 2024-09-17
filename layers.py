import math
import warnings
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from mmcv.cnn import CONV_LAYERS
from mmcv.cnn import build_norm_layer, build_padding_layer
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.utils import constant_init, kaiming_init


@CONV_LAYERS.register_module
class Wsc2d(pt.nn.Conv2d):

    def forward(self, input):
        weight = self.standardize_weight(self.weight)
        if self.padding_mode != 'zeros':
            return ptnf.conv2d(ptnf.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight,
                self.bias, self.stride, _pair(0), self.dilation, self.groups)
        return ptnf.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @staticmethod
    def standardize_weight(weight, eps=1e-5):
        mean = weight.mean([1, 2, 3], keepdim=True)
        std = weight.std([1, 2, 3], keepdim=True)
        return (weight - mean) / (std + eps)


@CONV_LAYERS.register_module
class Coc2d(pt.nn.Module):

    def __init__(self,
            in_channels: int, out_channels: int, kernel_size: _size_2_t,
            stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
            sa: int = 2,  # spatial-association: 2|3|4|5|6|7|8...
            sk_plus: int = 2,  # super-kernel-plus: kernel_size + 0|2|4...
            skip: bool = True,  # to alleviate over-smoothing
            dilat: bool = True,  # to alleviate over-smoothing
            wsc: bool = False,  # use Wsc2d as the super conv
            share: str = 'ci',  # share super conv's kernels on dim: ``ci``|``co``|``cico``
            ws: bool = False,  # do weight-standardization on the final weights T/F|``BN``|``LN``|``IN``
            both: bool = False,  # still use ``bn`` in super conv when ``wsc=True``  # TODO XXX remove
            sbn: str = 'BN',  # ``None``|``BN``|``LN``|``IN``  # TODO XXX
            c1x1_ws: bool = False,  # replace c1x1 with Wsc2d, for my ``build_conv_layer``
    ):
        assert kernel_size != 1
        super(Coc2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.transposed = None
        self.output_padding = None

        self.weight = pt.Tensor(out_channels, in_channels // groups, *kernel_size)  # XXX pt.nn.Parameter()
        if bias:
            self.bias = pt.nn.Parameter(pt.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        assert sa > 1
        assert sk_plus >= 0 and sk_plus % 2 == 0
        if dilat:  # to support c1x1; but replaceing-c1x1-with-Coc2d always degrades the acc
            dilat = not any([_ == 1 for _ in self.kernel_size])
            assert sk_plus > 0
        assert share in ['ci', 'co', 'cico', None]

        self.sa = sa
        self.skip = skip
        self.share = share
        self.ws = ws

        self.flag = self.weight.shape[0] % self.sa ** 2 != 0
        if self.flag:
            self.co = math.ceil(self.weight.shape[0] / self.sa ** 2) * self.sa ** 2
            _wght = pt.zeros(self.co - self.weight.shape[0], *self.weight.shape[1:], dtype=self.weight.dtype)
            self.weight1 = pt.cat([self.weight, _wght], dim=0)
        else:
            self.co = self.weight.shape[0]
            self.weight1 = self.weight
        self.co4 = self.co // self.sa ** 2
        self.ci = self.weight.shape[1]
        self.kh, self.kw = self.kernel_size

        if not dilat:
            sk, sd = tuple([_ + sk_plus for _ in self.kernel_size]), _pair(1)
        else:
            sk, sd = list(zip(*[self._expand_to_dilat(_, sk_plus, True) for _ in self.kernel_size]))

        sp = tuple([(_ // 2 * 2 * __ + 1) // 2 for _, __ in zip(sk, sd)])
        cv_cfg = dict(type='Wsc2d') if wsc else None
        # bn_cfg = dict(type='BN', requires_grad=True) if both or not wsc else None
        bn_cfg = dict(type=sbn, requires_grad=True) if both or not wsc else None  # TODO XXX
        if sbn == 'LN':
            assert share == 'ci', 'other share modes are not supported yet'
            bn_cfg.update({'normalized_shape': [self.co4, self.sa * self.kh, self.sa * self.kw]})

        if self.ws == 'BN':
            self.norm_final = pt.nn.BatchNorm2d(self.co)
        elif self.ws == 'LN':
            self.norm_final = pt.nn.LayerNorm([self.ci, self.kh, self.kw])
        elif self.ws == 'LayerNorm2d':
            self.norm_final = LayerNorm2d(self.ci)
        elif self.ws == 'IN':
            self.norm_final = pt.nn.InstanceNorm2d(self.ci, affine=True)

        # rearrange(self.weight, '(co_4 sa1 sa2) ci kh kw -> co_4 ci (sa1 kh) (sa2 kw)',  # not support fp16
        #     sa1=self.sa, sa2=self.sa)
        self.weight2 = self.weight1.view(self.co4, self.sa, self.sa, self.ci, self.kh, self.kw) \
            .permute(0, 3, 1, 4, 2, 5).contiguous() \
            .view(self.co4, self.ci, self.sa * self.kh, self.sa * self.kw)
        if self.share is None:
            ci_co_sk_ss_sp_sd_sg = (self.ci, self.ci, sk, 1, sp, sd, 1)
            self.weight3 = self.weight2
        elif self.share == 'ci':
            ci_co_sk_ss_sp_sd_sg = (self.co4, self.co4, sk, 1, sp, sd, self.co4)
            self.weight3 = self.weight2.permute(1, 0, 2, 3)
        elif self.share == 'co':
            ci_co_sk_ss_sp_sd_sg = (self.ci, self.ci, sk, 1, sp, sd, self.ci)
            self.weight3 = self.weight2
        elif self.share == 'cico':
            ci_co_sk_ss_sp_sd_sg = (1, 1, sk, 1, sp, sd, 1)
            self.weight3 = self.weight2.view(self.co4 * self.ci, 1, self.sa * self.kh, self.sa * self.kw)
        else:
            raise NotImplemented
        self.weight9 = pt.nn.Parameter(self.weight3)

        self.conv_super = ConvModule(*ci_co_sk_ss_sp_sd_sg, conv_cfg=cv_cfg, norm_cfg=None, act_cfg=None)
        self.reset_parameters()

    def forward(self, xi):
        self_weight_conv = self.conv_super(self.weight9)
        if self.skip:
            self_weight_conv = self_weight_conv + self.weight9

        if self.share is None:
            pass
        elif self.share == 'ci':
            self_weight_conv = self_weight_conv.permute(1, 0, 2, 3)
        elif self.share == 'co':
            pass
        elif self.share == 'cico':
            self_weight_conv = self_weight_conv.view(self.co4, self.ci, self.sa * self.kh, self.sa * self.kw)
        else:
            raise NotImplemented

        self_weight_reshape = self_weight_conv.view(self.co4, self.ci, self.sa, self.kh, self.sa, self.kw) \
            .permute(0, 2, 4, 1, 3, 5).contiguous() \
            .view(self.co4 * self.sa * self.sa, self.ci, self.kh, self.kw)
        if self.flag:
            self_weight_reshape = self_weight_reshape[:self.weight.shape[0]]

        if self.ws in [True, False]:
            if self.ws:
                self_weight = Wsc2d.standardize_weight(self_weight_reshape)
            else:
                self_weight = self_weight_reshape
        elif self.ws in ['BN', 'LN', 'LayerNorm2d', 'IN']:
            self_weight = self.norm_final(self_weight_reshape)
        else:
            raise NotImplemented
        xo = ptnf.conv2d(xi, self_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return xo

    @staticmethod
    def _expand_to_dilat(k0, ex, larger):
        rows = [3, 5, 7]
        cols = [0, 2, 4, 6]
        assert k0 in rows and ex in cols
        if not larger:
            mat = np.array([
                [[3, 1], [3, 2], [3, 3], [5, 2]],
                [[3, 2], [3, 3], [5, 2], [5, 3]],
                [[3, 3], [5, 2], [5, 3], [5, 4]]
            ], dtype=np.int)
        else:
            mat = np.array([
                [[3, 1], [3, 2], [3, 3], [5, 2]],
                [[3, 3], [5, 2], [5, 3], [5, 4]],
                [[5, 2], [5, 3], [5, 4], [7, 3]]
            ], dtype=np.int)
        sk, sd = mat[rows.index(k0), cols.index(ex)]
        return sk, sd

    def reset_parameters(self):
        pt.nn.init.kaiming_uniform_(self.weight9, a=math.sqrt(5))  # XXX self.weight
        if self.bias is not None:
            fan_in, _ = pt.nn.init._calculate_fan_in_and_fan_out(self.weight9)  # XXX self.weight
            bound = 1 / math.sqrt(fan_in)
            pt.nn.init.uniform_(self.bias, -bound, bound)
        # XXX init of super layers uses ``ResNet.init_weights()``


@CONV_LAYERS.register_module
class LayerNorm2d(pt.nn.LayerNorm):

    def __init__(self, num_channels, eps: float = 1e-5, elementwise_affine: bool = True):
        super(LayerNorm2d, self).__init__([num_channels], eps, elementwise_affine)

    def forward(self, input: pt.Tensor) -> pt.Tensor:
        xi = input.permute(0, 2, 3, 1)
        x1 = pt.nn.functional.layer_norm(xi, self.normalized_shape, self.weight, self.bias, self.eps)
        x2 = x1.permute(0, 3, 1, 2)
        return x2


def build_conv_layer(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type == 'Coc2d':
        if (len(args) >= 3 and args[2] == 1) or ('kernel_size' in kwargs and kwargs.get('kernel_size') == 1):
            if cfg.get('c1x1_ws', False):
                layer_type = 'Wsc2d'  # XXX replace c1x1 with Wsc2d
            else:
                layer_type = 'Conv2d'  # XXX just replace c1x1 with Coc2d
            cfg_ = dict()

    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


class ConvModule(nn.Module):
    _abbr_ = 'conv_block'

    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias='auto',
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=dict(type='ReLU'),
            inplace=True,
            with_spectral_norm=False,
            padding_mode='zeros',
            order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            if norm_cfg['type'] == 'LN':  # TODO XXX to support ``LN``
                norm_channels = norm_cfg['normalized_shape']
                norm_cfg.pop('normalized_shape')
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x
