# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# 架构字符串到真实模型转换之间所需要的各种类

# 复合层，包含 ReLU 激活、卷积和批量归一化。
class ReLUConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, affine, track_running_stats=True, use_bn=True, name='ReLUConvBN'):
        super(ReLUConvBN, self).__init__()
        self.name = name
        # 判断是否启用批量归一化层
        if use_bn:
            self.op = nn.Sequential(
                # 将输入张量中所有的负值设为 0，而保持正值不变。inplace=False：表示 ReLU 激活不会直接修改输入张量，而是会创建一个新的张量来存储激活结果。
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not affine),
                nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=track_running_stats)
                )
        else:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not affine)
                )

    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    def __init__(self, name='Identity'):
        self.name = name
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):

  def __init__(self, stride, name='Zero'):
    self.name = name
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

class POOLING(nn.Module):
    def __init__(self, kernel_size, stride, padding, name='POOLING'):
        super(POOLING, self).__init__()
        self.name = name
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.avgpool(x)

# 用于下采样输入特征图，并增加通道数。下采样：减少特征图或数据的分辨率，即降低其宽度和高度（空间维度）或减少数据点数量。在深度学习中，下采样主要用于降低数据的空间尺寸，以减少计算量并逐步提取更高层次的特征。
class reduction(nn.Module):
    # 输出特征图的通道数，通常会设置为输入通道数的倍数，用于增加特征的表达能力。
    def __init__(self, in_channels, out_channels):
        super(reduction, self).__init__()
        self.residual = nn.Sequential(
                            # 平均池化层，步幅为 2，作用是将特征图的宽和高各缩小一半（下采样）
                            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                            # 将池化后的特征图通道数从 in_channels 转换为 out_channels，通常用于调整特征图的通道数。
                            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        # 将输入张量中所有的负值设为 0，而保持正值不变。
        self.conv_a = ReLUConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, dilation=1, affine=True, track_running_stats=True)
        # 步幅为 1，用于进一步处理特征图，但不会改变尺寸和通道数。
        self.conv_b = ReLUConvBN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, affine=True, track_running_stats=True)

    def forward(self, x):
        basicblock = self.conv_a(x)
        basicblock = self.conv_b(basicblock) # 主路径
        residual = self.residual(x) # 残差路径
        # 主路径和残差路径相加后，既保留了输入特征信息，又能通过主路径提取新的特征。这一设计尤其适用于下采样操作，因为它可以在减小特征图尺寸的同时，确保信息传递的完整性，从而增强模型的表达能力。
        return residual + basicblock

# 定义了一个卷积神经网络的第一层（stem 层），主要用于对输入的图像进行初步特征提取。
class stem(nn.Module):
    def __init__(self, out_channels, use_bn=True):
        super(stem, self).__init__()
        # 控制是否在卷积层后添加批量归一化
        if use_bn:
            self.net = nn.Sequential(
                    # Conv2d 是用于处理图像等网格结构数据的卷积层，输入通常是四维张量 [batch_size, channels, height, width]
                    nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels))
        else:
            self.net = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        return self.net(x)

# 输出层模块，通常用于分类任务的最后部分。
class top(nn.Module):
    def __init__(self, in_dims, num_classes, use_bn=True):
        super(top, self).__init__()
        # 归一化和激活层
        if use_bn:
            self.lastact = nn.Sequential(nn.BatchNorm2d(in_dims), nn.ReLU(inplace=True))
        else:
            self.lastact = nn.ReLU(inplace=True)
        # 全局池化层，使用自适应平均池化，将输入特征图的宽和高缩小到 1x1，使每个通道只有一个数值。
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # 分类层，将池化后的特征映射到分类输出上，用于最终的分类任务。
        self.classifier  = nn.Linear(in_dims, num_classes)

    def forward(self, x):
        x = self.lastact(x)
        # 将特征图通过全局池化，缩小到 [batch_size, in_dims, 1, 1]，将空间信息压缩为单个特征值。
        x = self.global_pooling(x)
        # 将 x 展平，使其变成 [batch_size, in_dims] 的形状，方便输入到全连接层。
        x = x.view(x.size(0), -1)
        # 通过全连接层得到 logits，即分类的输出，每个类别对应一个得分。
        logits = self.classifier(x)
        return logits, x

# 用于神经架构搜索（NAS）过程中的基本单元
class SearchCell(nn.Module):
    # affine 和 track_running_stats：用于批量归一化层的配置，控制是否使用仿射变换和是否跟踪均值、方差
    def __init__(self, in_channels, out_channels, stride, affine, track_running_stats, use_bn=True, num_nodes=4, keep_mask=None):
        super(SearchCell, self).__init__()
        #  SearchCell 的结构中包含多少个节点
        self.num_nodes = num_nodes
        # 用于存储节点之间的可选操作
        self.options = nn.ModuleList()
        # 对于所有节点curr_node
        for curr_node in range(self.num_nodes-1):
            # 遍历curr_node之前所有的节点，注意range的性质导致+1
            for prev_node in range(curr_node+1):
                # 对于每一个前后连接，遍历所有可用的操作
                for _op_name in OPS.keys():
                    # 基于传入的参数创建每种操作的实例
                    op = OPS[_op_name](in_channels, out_channels, stride, affine, track_running_stats, use_bn)
                    # self.options 最终会存储当前节点和前一个节点之间所有可能的连接操作
                    self.options.append(op)

        # 用来控制哪些操作连接是激活的、哪些是被忽略的。
        if keep_mask is not None:
            self.keep_mask = keep_mask
        # 默认设置所有操作连接为保留状态
        else:
            self.keep_mask = [True]*len(self.options)

    # 根据 self.keep_mask 控制哪些操作连接是激活的、哪些被忽略，从而灵活地构建网络的前向传播路径
    def forward(self, x):
        # 初始化一个列表 outs，用于存储每个节点的结果。初始值为输入 x，代表节点 0 。
        outs = [x]

        idx = 0
        # 遍历每个节点
        for curr_node in range(self.num_nodes-1):
            # edges_in 中包含了当前节点从所有前驱节点接收到的有效特征输出。
            edges_in = []
            # 遍历当前节点的所有前驱节点
            for prev_node in range(curr_node+1):
                # 遍历所有可能的操作，以便逐个检查 keep_mask 是否为 True。
                for op_idx in range(len(OPS.keys())):
                    if self.keep_mask[idx]:
                        # 将 outs[prev_node]（前一个节点的输出）作为输入，执行对应的操作实例，并将结果存入 edges_in。
                        edges_in.append(self.options[idx](outs[prev_node]))
                    idx += 1
            # 使用求和聚合来融合来自不同路径的信息，生成当前节点的聚合输出 node_output。
            node_output = sum(edges_in)
            outs.append(node_output)

        # 返回的是 outs 列表的最后一个元素，即最后一个节点的输出，因为在前向传播的过程中，我们通常只关心最后一个节点的输出。
        return outs[-1]


# lambda 函数可以动态接收参数并返回对应的操作实例，而不需要单独定义多个函数。
OPS = {
    'none' : lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: Zero(stride, name='none'),
    'avg_pool_3x3' : lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: POOLING(3, 1, 1, name='avg_pool_3x3'),
    'nor_conv_3x3' : lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: ReLUConvBN(in_channels, out_channels, 3, 1, 1, 1, affine, track_running_stats, use_bn, name='nor_conv_3x3'),
    'nor_conv_1x1' : lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: ReLUConvBN(in_channels, out_channels, 1, 1, 0, 1, affine, track_running_stats, use_bn, name='nor_conv_1x1'),
    'skip_connect' : lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: Identity(name='skip_connect'),
}


