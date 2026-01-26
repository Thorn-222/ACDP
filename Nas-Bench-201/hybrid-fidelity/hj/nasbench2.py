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

#架构字符串和对应的神经网络架构之间的转换！


import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from nasbench2_ops import *

# 获取架构特征
def gen_searchcell_mask_from_arch_str(arch_str):
    # 将架构字符串 arch_str 按 '+' 分割成节点
    nodes = arch_str.split('+')
    # 去除括号并按 '|' 分割，进一步分割节点内部结构
    nodes = [node[1:-1].split('|') for node in nodes]
    # 将节点中每个操作（例如 nor_conv_3x3~0）拆分成操作类型和输入节点
    nodes = [[op_and_input.split('~')  for op_and_input in node] for node in nodes]
    '''
    假设 arch_str 是以下字符串：
    "|nor_conv_3x3~0|+|none~0|nor_conv_1x1~1|+|avg_pool_3x3~0|nor_conv_3x3~1|none~2|"
    1、按 '+' 分割，得到三个节点的字符串：
    nodes = ["|nor_conv_3x3~0|", "|none~0|nor_conv_1x1~1|", "|avg_pool_3x3~0|nor_conv_3x3~1|none~2|"]
    2、去除每个节点的首尾括号并按 '|' 分割：
    nodes = [["nor_conv_3x3~0"], ["none~0", "nor_conv_1x1~1"], ["avg_pool_3x3~0", "nor_conv_3x3~1", "none~2"]]
    3、进一步按 '~' 分割每个操作：
    nodes = [
    [["nor_conv_3x3", "0"]],
    [["none", "0"], ["nor_conv_1x1", "1"]],
    [["avg_pool_3x3", "0"], ["nor_conv_3x3", "1"], ["none", "2"]]
    ]
    '''
    # 用于存储布尔值列表，表示是否保留每个操作
    keep_mask = []
    # 遍历 nodes 中的每个节点，curr_node_idx 表示当前节点的索引，依次检查每个节点的操作，并生成其与前一个节点的连接情况。
    for curr_node_idx in range(len(nodes)):
            # 遍历 curr_node_idx 之前的所有节点，prev_node_idx 表示前一个节点的索引
            for prev_node_idx in range(curr_node_idx+1):
                # 从 nodes[curr_node_idx] 中找到所有输入来自 prev_node_idx 的操作
                _op = [edge[0] for edge in nodes[curr_node_idx] if int(edge[1]) == prev_node_idx]
                # _op 列表中只能有一个操作类型。如果发现有多个操作，抛出异常，表明架构字符串不符合假设。
                assert len(_op) == 1, 'The arch string does not follow the assumption of 1 connection between two nodes.'
                # 逐个检查 _op[0]（当前操作）是否匹配 OPS 中的每一个操作
                for _op_name in OPS.keys():
                    keep_mask.append(_op[0] == _op_name)
    return keep_mask
    ''''''
    '''
    keep_mask 将包含 OPS 中每种操作在指定节点对之间是否存在的布尔值。假设得到的 keep_mask 是：
    [
    True, False, False, False,
    False, True, False, False,
    False, False, True, False,
    False, False, False, True,
    True, False, False, False,
    False, True, False, False
    ]
    第 1 行：表示从第 0 个节点到第 1 个节点的操作。
    第 2 行：表示从第 0 个节点到第 2 个节点的操作。
    第 3 行：表示从第 1 个节点到第 2 个节点的操作。
    第 4 行：表示从第 0 个节点到第 3 个节点的操作。
    第 5 行：表示从第 1 个节点到第 3 个节点的操作。
    第 6 行：表示从第 2 个节点到第 3 个节点的操作。
    '''

# 根据给定的架构字符串 arch_str 创建一个特定的 NAS 模型实例
def get_model_from_arch_str(arch_str, num_classes, use_bn=True, init_channels=16):
    # 生成连接掩码 keep_mask
    keep_mask = gen_searchcell_mask_from_arch_str(arch_str)
    net = NAS201Model(arch_str=arch_str, num_classes=num_classes, use_bn=use_bn, keep_mask=keep_mask, stem_ch=init_channels)
    return net

# 根据架构字符串生成对应的神经网络架构，然后可以对这个实际网络进行训练，从而得到该架构在数据集上的真实性能值。
class NAS201Model(nn.Module):

    def __init__(self, arch_str, num_classes, use_bn=True, keep_mask=None, stem_ch=16):
        super(NAS201Model, self).__init__()
        # 模型架构的字符串
        self.arch_str=arch_str
        # 指定分类任务的类别数量，决定了输出层的维度，架构训练的目的就是为了输出图片属于什么类别，故输出层的维度=类别数
        self.num_classes=num_classes
        # 表示是否使用批量归一化
        self.use_bn= use_bn

        # 第一个卷积层（stem 层），将输入的图像数据转换为具有 stem_ch 个通道的特征图。
        self.stem = stem(out_channels=stem_ch, use_bn=use_bn)
        # 第一个堆叠单元层，由 5 个 SearchCell 组成，每个单元的输入和输出通道数为 stem_ch。
        self.stack_cell1 = nn.Sequential(*[SearchCell(in_channels=stem_ch, out_channels=stem_ch, stride=1, affine=False, track_running_stats=False, use_bn=use_bn, keep_mask=keep_mask) for i in range(5)])
        # 下采样输入特征图，并增加通道数。
        self.reduction1 = reduction(in_channels=stem_ch, out_channels=stem_ch*2)
        # 第二个堆叠单元
        self.stack_cell2 = nn.Sequential(*[SearchCell(in_channels=stem_ch*2, out_channels=stem_ch*2, stride=1, affine=False, track_running_stats=False, use_bn=use_bn, keep_mask=keep_mask) for i in range(5)])
        self.reduction2 = reduction(in_channels=stem_ch*2, out_channels=stem_ch*4)#  out_channels=stem_ch*4,512的地方都是
        # 第三个堆叠单元
        self.stack_cell3 = nn.Sequential(*[SearchCell(in_channels=stem_ch*4, out_channels=stem_ch*4, stride=1, affine=False, track_running_stats=False, use_bn=use_bn, keep_mask=keep_mask) for i in range(5)])
        # 输出层
        self.top = top(in_dims=stem_ch*4, num_classes=num_classes, use_bn=use_bn)

    def forward(self, x):
        x = self.stem(x)        

        x = self.stack_cell1(x)
        x = self.reduction1(x)

        x = self.stack_cell2(x)
        x = self.reduction2(x)

        x_features = self.stack_cell3(x)

        x,x_features_pooled = self.top(x_features)
        return x, x_features, x_features_pooled

    # 生成模型的一个可剪枝副本，通常在神经网络的剪枝操作中使用。剪枝是一种减少模型参数的方法，用于减少计算量或内存占用，而该方法创建的副本则是在剪枝前的初始化模型。
    def get_prunable_copy(self, bn=False):
        # 创建一个与当前模型结构相同的模型副本。但在创建时批量归一化的使用情况可能不同，bn 控制是否在新模型中启用批量归一化
        model_new = get_model_from_arch_str(self.arch_str, self.num_classes, use_bn=bn)

        #TODO this is quite brittle and doesn't work with nn.Sequential when bn is different
        # it is only required to maintain initialization -- maybe init after get_punable_copy?
        # 建议将剪枝的初始化放在 get_prunable_copy 调用之后，以便保持模型的初始化状态。

        # 将当前模型的权重和偏置复制到新的模型 model_new 中。
        model_new.load_state_dict(self.state_dict(), strict=False)
        # 将 model_new 设置为训练模式，这样它会在后续剪枝或微调操作中保持正确的行为（如启用批量归一化和 dropout）
        model_new.train()

        return model_new
    
# 从模型 net 中提取架构字符串（arch_str）
def get_arch_str_from_model(net):
    # net.stack_cell1[0] 表示 stack_cell1 中第一个 SearchCell 单元。
    # 已经包含了整个网络架构的基本信息，包括节点、连接方式和操作类型。所以仅从一个单元就能推断出整个架构字符串。
    search_cell = net.stack_cell1[0].options
    keep_mask = net.stack_cell1[0].keep_mask
    num_nodes = net.stack_cell1[0].num_nodes

    nodes = []
    idx = 0
    # 对于每一个节点
    for curr_node in range(num_nodes -1):
        edges = []
        # 遍历当前节点的所有前驱节点
        for prev_node in range(curr_node+1):
            # 对于每一个可能的操作（即keep_mask中一行中的所有布尔值）
            for _op_name in OPS.keys():
                # 存在当前操作
                if keep_mask[idx]:
                    edges.append(f'{_op_name}~{prev_node}')
                idx += 1
        node_str = '|'.join(edges)
        node_str = f'|{node_str}|'
        nodes.append(node_str) 
    arch_str = '+'.join(nodes)
    return arch_str


if __name__ == "__main__":
    arch_str = '|nor_conv_3x3~0|+|none~0|none~1|+|avg_pool_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|'
    
    n = get_model_from_arch_str(arch_str=arch_str, num_classes=10)
    print(n.stack_cell1[0])
    
    arch_str2 = get_arch_str_from_model(n)
    print(arch_str)
    print(arch_str2)
    print(f'Are the two arch strings same? {arch_str == arch_str2}')
