import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.weight_init import caffe2_xavier_init
from torch.utils.checkpoint import checkpoint

from mmdet.ops import ConvModule
from ..registry import NECKS


@NECKS.register_module
class HRFPN(nn.Module):
    """HRFPN (High Resolution Feature Pyrmamids)

    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        stride (int): stride of 3x3 convolutional layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False,
                 stride=1,
                 outlayer=None):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.out_layer = outlayer

        self.reduction_conv = ConvModule(
            sum(in_channels),
            out_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            act_cfg=None)

        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    act_cfg=None))

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

        self.extract_context = False
        # if self.extract_context:
        #     self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #     self.max_pool = nn.AdaptiveMaxPool2d(1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        # origin
        if self.out_layer is None:
            assert len(inputs) == self.num_ins
            outs = [inputs[0]]
            for i in range(1, self.num_ins):
                outs.append(
                    F.interpolate(inputs[i], scale_factor=2 ** i, mode='bilinear'))
            out = torch.cat(outs, dim=1)
            # pdb.set_trace()
            out_cat = out.clone()
            if out.requires_grad and self.with_cp:
                out = checkpoint(self.reduction_conv, out)
            else:
                out = self.reduction_conv(out)
            outs = [out]
            for i in range(1, self.num_outs):
                outs.append(self.pooling(out, kernel_size=2 ** i, stride=2 ** i))
            outputs = []

            for i in range(self.num_outs):
                if outs[i].requires_grad and self.with_cp:
                    tmp_out = checkpoint(self.fpn_convs[i], outs[i])
                else:
                    tmp_out = self.fpn_convs[i](outs[i])
                outputs.append(tmp_out)
            # pdb.set_trace()
            if self.extract_context:
                return tuple([out_cat, outputs])
            else:
                return tuple(outputs)
            # return tuple([outputs[0]])
        else:
            # pdb.set_trace()
            assert len(inputs) == self.num_ins
            idx = self.out_layer
            outputs = []
            for j in range(self.num_ins):
                if j<idx:
                    out_i=inputs[j]
                    for i in range(j,idx):
                        out_i=F.avg_pool2d(out_i, kernel_size=2, stride=2)
                    outputs.append(out_i)
                if j==idx:
                    outputs.append(inputs[j])
                if j>idx:
                    outputs.append(F.interpolate(inputs[j], scale_factor=2**(j-idx), mode='bilinear'))
            # pdb.set_trace()
            out = torch.cat(outputs, dim=1)
            # pdb.set_trace()
            if out.requires_grad and self.with_cp:
                out = checkpoint(self.reduction_conv, out)
            else:
                out = self.reduction_conv(out)
            outs=[out]
            if self.num_outs>1:
                for i in range(1, self.num_outs):
                    # pooling
                    outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
                    # dilated conv
                    # outs.append(self.fpn_dilatedconvs[i-1](outs[-1]))
            # pdb.set_trace()
            outputs = []
            for i in range(self.num_outs):
                if outs[i].requires_grad and self.with_cp:
                    tmp_out = checkpoint(self.fpn_convs[i], outs[i])
                else:
                    tmp_out = self.fpn_convs[i](outs[i])
                outputs.append(tmp_out)
            # pdb.set_trace()
            return tuple(outputs)
