# full connected like hrnet

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..registry import BACKBONES
from mmdet.ops import build_conv_layer, build_norm_layer
from .resnet import BasicBlock, Bottleneck
import pdb, json

# soft gate for path choice
def soft_gate(x, x_t=None, momentum=0.1, is_update=False):
    if is_update:
        # using momentum for weight update
        y = (1 - momentum) * x.data + momentum * x_t     # 指数移动平均
        tanh_value = torch.tanh(y)
        return F.relu(tanh_value), y.data
    else:
        # tanh_value = torch.tanh(x)
        # return F.relu(tanh_value)
        sig_value = torch.sigmoid(x)
        return sig_value


class DynamicHRModule(nn.Module):
    """ High-Resolution Module for HRNet. In this module, every branch
    has 4 BasicBlocks/Bottlenecks. Fusion/Exchange is in this module.
    """

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=True,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 num_backbones=1):
        super(DynamicHRModule, self).__init__()
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.gate_layers = self._make_dynamic_gate_layers(num_backbones)
        self.relu = nn.ReLU(inplace=False)
        self.num_backbones=num_backbones

    def _check_branches(self, num_branches, num_blocks, in_channels,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(in_channels))
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, num_channels[branch_index] *
                                 block.expansion)[1])

        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_dynamic_gate_layers(self, num_backbones=1):
        if self.num_branches<4:
            return None

        num_branches = self.num_branches
        gate_num = num_branches
        in_channels = self.in_channels
        gate_layers_backone = []
        for num_backbone in range(num_backbones):
            gate_layers=[]
            for i in range(num_branches):
                # if i==0 or i==num_branches-1:   # only keep and down
                #     gate_num=2
                # else:
                #     gate_num=3
                gate_layers.append(
                    nn.Sequential(
                        build_conv_layer(
                            self.conv_cfg,
                            in_channels[i],
                            in_channels[i]//2,  # //2
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False),
                        build_norm_layer(self.norm_cfg,
                                         in_channels[i]//2)[1],   # //2
                        nn.ReLU(inplace=False),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        build_conv_layer(
                            self.conv_cfg,
                            in_channels[i]//2,   # //2
                            gate_num,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True))
                )
            gate_layers_backone.append(nn.ModuleList(gate_layers))
        return nn.ModuleList(gate_layers_backone)


    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):    # i指融合后，j指融合前
            fuse_layer = []
            for j in range(num_branches):
                if j > i:      # resolution-up and dim-down 1x1降维再升高分辨率
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i), mode='nearest')))
                elif j == i:   # 同层直接运算
                    fuse_layer.append(None)
                else:  # j<i
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(    # 3x3 conv+stride=2
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],  # downsample but dim not change -> lose some info
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)


    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        backbone_idx=x[1]
        x=x[0]
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # pdb.set_trace()
        if backbone_idx is None: # or (self.num_branches<dynamic):
            for i in range(self.num_branches):
                x[i] = self.branches[i](x[i])

            x_fuse = []
            for i in range(len(self.fuse_layers)):
                y = 0
                for j in range(self.num_branches):
                    if i == j:
                        y += x[j]
                    else:
                        y += self.fuse_layers[i][j](x[j])
                x_fuse.append(self.relu(y))
            return [x_fuse, backbone_idx]
        else:

            gate_weights_beta=[]
            gate_masks=[]
            for i in range(self.num_branches):
                x[i] = self.branches[i](x[i])
                gate_feat_beta = self.gate_layers[backbone_idx][i](x[i])  # (1, gate_num, 1, 1)
                gate_weight_beta = soft_gate(gate_feat_beta)
                gate_mask = (gate_weight_beta.sum(dim=1, keepdim=True) < 0.0001).float()
                gate_weights_beta.append(gate_weight_beta)
                gate_masks.append(gate_mask)

            # save gate weights
            # weights=[]
            # for weight in gate_weights_beta:
            #     weight = weight.squeeze().cpu().tolist()
            #     weight=[round(w,4) for w in weight]
            #     weights.append(weight)
            #
            # weight_backbone='backbone'+str(backbone_idx)
            # weight_stage='stage'+str(self.num_branches)
            # key = weight_stage+'_'+weight_backbone
            # # pdb.set_trace()
            # weights={key: weights}
            # with open('analyse/dynamic_gate/val_cropImgs_dynamicFromS4_gateWeights_lr=0002_sigmoidfanin.json', 'a') as f:
            #     json.dump(weights, f)
            #     f.write(',')

            x_fuse = []
            zero_label = []
            for i in range(len(self.fuse_layers)):
                y = 0
                zero_label.append(0)
                for j in range(self.num_branches):
                    # todo: fully connected
                    weight = gate_weights_beta[j][:,i].unsqueeze(-1)
                    if i == j:
                        y += (x[j]*gate_masks[j] + x[j]*weight)  # if weight is too small, keep original feature
                    else:
                        y += (self.fuse_layers[i][j](x[j]))*weight

                x_fuse.append(self.relu(y))
            return [x_fuse, backbone_idx]


@BACKBONES.register_module
class DynamicHRNet(nn.Module):
    """HRNet backbone.

    High-Resolution Representations for Labeling Pixels and Regions
    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Normally 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import HRNet
        >>> import torch
        >>> extra = dict(
        >>>     stage1=dict(
        >>>         num_modules=1,
        >>>         num_branches=1,
        >>>         block='BOTTLENECK',
        >>>         num_blocks=(4, ),
        >>>         num_channels=(64, )),
        >>>     stage2=dict(
        >>>         num_modules=1,
        >>>         num_branches=2,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4),
        >>>         num_channels=(32, 64)),
        >>>     stage3=dict(
        >>>         num_modules=4,
        >>>         num_branches=3,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4),
        >>>         num_channels=(32, 64, 128)),
        >>>     stage4=dict(
        >>>         num_modules=3,
        >>>         num_branches=4,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4, 4),
        >>>         num_channels=(32, 64, 128, 256)))
        >>> self = HRNet(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 32, 8, 8)
        (1, 64, 4, 4)
        (1, 128, 2, 2)
        (1, 256, 1, 1)
    """

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=False,
                 num_backbones=1):
        super(DynamicHRNet, self).__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.num_backbones=num_backbones
        # pdb.set_trace()
        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:  # stage1->2
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                DynamicHRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    num_backbones=self.num_backbones))

        return nn.Sequential(*hr_modules), in_channels

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
            # todo: init weight for gates
            for m in self.modules():
                if isinstance(m, DynamicHRModule):
                    if 'gate_layers' in m.__dict__['_modules'].keys():
                        # pdb.set_trace()
                        m_gate = m.__dict__['_modules']['gate_layers']
                        for m_g in m_gate.modules():
                            # pdb.set_trace()
                            if isinstance(m_g, nn.Conv2d):
                                kaiming_init(m_g, mode='fan_in', bias=2.2)   # 1.5
                            elif isinstance(m_g, (_BatchNorm, nn.GroupNorm)):
                                constant_init(m_g, 1)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.layer1(x)    # stage1

        # pdb.set_trace()
        y_list_backbone = []
        # # for num_backbone in range(self.num_backbones):
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        # pdb.set_trace()
        y_list,_ = self.stage2([x_list,None])

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        # pdb.set_trace()
        y_list,_ = self.stage3([x_list,None])


        for num_backbone in range(self.num_backbones):
            # x_list = []
            # for i in range(self.stage2_cfg['num_branches']):
            #     if self.transition1[i] is not None:
            #         x_list.append(self.transition1[i](x))
            #     else:
            #         x_list.append(x)
            # # pdb.set_trace()
            # y_list, _ = self.stage2([x_list, num_backbone])
            #
            # x_list = []
            # for i in range(self.stage3_cfg['num_branches']):
            #     if self.transition2[i] is not None:
            #         x_list.append(self.transition2[i](y_list[-1]))
            #     else:
            #         x_list.append(y_list[i])
            # # pdb.set_trace()
            # y_list_, _ = self.stage3([x_list, num_backbone])

            x_list = []
            for i in range(self.stage4_cfg['num_branches']):
                if self.transition3[i] is not None:
                    x_list.append(self.transition3[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list_,_ = self.stage4([x_list,num_backbone])
            y_list_backbone.append(y_list_)
        # pdb.set_trace()
        return y_list_backbone

    def _freeze_stages(self):
        for m in self.modules():
            if isinstance(m, DynamicHRModule):
                # pdb.set_trace()
                for key in m.__dict__['_modules'].keys():
                    if key=='gate_layers':
                        continue
                    else:
                        m_gate = m.__dict__['_modules'][key]
                        for m_g in m_gate.modules():
                            if isinstance(m_g, (nn.Conv2d, _BatchNorm, nn.GroupNorm)):
                                m_g.eval()
                                for param in m_g.parameters():
                                    param.requires_grad = False
            elif isinstance(m, (nn.Conv2d,_BatchNorm, nn.GroupNorm)):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(DynamicHRNet, self).train(mode)
        # === todo ======
        # self._freeze_stages()
        # ===================
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
