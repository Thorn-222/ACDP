
from __future__ import print_function
import os
import time
import igraph
import random
import numpy as np
import scipy.stats
import argparse
import torch


def load_graph_config(graph_data_name, nvt, data_path):
  if graph_data_name is not 'nasbench201':
    raise NotImplementedError(graph_data_name)
  g_list = []
  max_n = 0
  ms = torch.load(os.path.join(data_path, f'{graph_data_name}.pt'))['arch']['matrix']
  # 处理每个图架构
  for i in range(len(ms)):
    # 将每个网络架构矩阵 ms[i] 解码成一个图（使用 igraph 数据结构）和节点数量 n
    g, n = decode_NAS_BENCH_201_8_to_igraph(ms[i])
    # 更新 max_n 变量，记录最大节点数
    max_n = max(max_n, n)
    g_list.append((g, 0))
  # number of different node types including in/out node
  graph_config = {}
  # 节点类型的数量
  graph_config['num_vertex_type'] = nvt # original types + start/end types
  graph_config['max_n'] = max_n # maximum number of nodes
  # 预定义的起始节点类型
  graph_config['START_TYPE'] = 0 # predefined start vertex type
  # 预定义的结束节点类型
  graph_config['END_TYPE'] = 1 # predefined end vertex type

  return graph_config

# 将一个表示神经网络结构（或图结构）的列表 row 转换成 igraph 图对象
def decode_NAS_BENCH_201_8_to_igraph(row):
  # 如果输入是一个字符串（如从文件读取的数据），eval(row) 会将这个字符串转换成一个列表
  if type(row) == str:
    row = eval(row)  # convert string to list of lists
  # 获取节点数
  n = len(row)
  # 初始化有向图
  g = igraph.Graph(directed=True)
  # 添加节点
  g.add_vertices(n)
  for i, node in enumerate(row):
    # 为每个节点设置节点类型（row每行的第一个元素代表该行对应节点的类型）
    g.vs[i]['type'] = node[0]
    # 如果当前节点不是最后两个节点，程序会在当前节点 i 和下一个节点 i + 1 之间添加一条边
    # 示在该数据结构中存在一个由节点顺序连接的链
    if i < (n - 2) and i > 0:
      g.add_edge(i, i + 1)  # always connect from last node
    # 根据节点连接信息添加更多边，如果 node[1:] 中某个元素为 1，那么就会在 i 节点和 j 节点之间添加一条边
    for j, edge in enumerate(node[1:]):
      if edge == 1:
        g.add_edge(j, i)
  return g, n

# 检查一个图 g 是否是一个有效的有向无环图 (DAG)
def is_valid_DAG(g, START_TYPE=0, END_TYPE=1):
  # igraph 库提供的方法，它会检查图是否是有向无环图
  res = g.is_dag()
  n_start, n_end = 0, 0
  for v in g.vs:
    if v['type'] == START_TYPE:
      n_start += 1
    elif v['type'] == END_TYPE:
      n_end += 1
    # 该节点没有任何输入边，但却不是起始节点，图不符合要求
    if v.indegree() == 0 and v['type'] != START_TYPE:
      return False
    # 该节点没有任何输出边，但却不是结束节点，图不符合要求
    if v.outdegree() == 0 and v['type'] != END_TYPE:
      return False
  return res and n_start == 1 and n_end == 1

# 检查一个图 g 是否符合 NAS-Bench-201 的规范，确保图符合特定条件
def is_valid_NAS201(g, START_TYPE=0, END_TYPE=1):
  # first need to be a valid DAG computation graph
  # 检查是否为有效的有向无环图 (DAG)，开始节点是否是 START_TYPE，结束节点是否是 END_TYPE
  res = is_valid_DAG(g, START_TYPE, END_TYPE)
  # in addition, node i must connect to node i+1
  # 要求图中必须包含 8 个节点
  res = res and len(g.vs['type'])==8
  # 确保在中间节点中不存在类型为 0 的节点
  res = res and not (0 in g.vs['type'][1:-1])
  # 确保在中间节点中不存在类型为 1 的节点
  res = res and not (1 in g.vs['type'][1:-1])
  # 结果是一个布尔值
  return res

# 将图 g 解码为一个 4x4 的矩阵 m
def decode_igraph_to_NAS201_matrix(g):
  m = [[0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0]]
  # 预定义一些坐标位置，表示图中某些节点的关系
  xys = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
  # 遍历这些坐标，在矩阵 m 的对应位置上填充节点对应的操作
  for i, xy in enumerate(xys):
    m[xy[0]][xy[1]] = float(g.vs[i + 1]['type']) - 2
  import numpy
  # 将矩阵转为 NumPy 数组并返回
  return numpy.array(m)


# 将图 g 解码为 NAS-Bench-201 的字符串表示（批量处理）
def decode_igraph_to_NAS_BENCH_201_string(gs):
  # 返回存储解码结果的列表
  result = []

  # 对每个图进行处理
  for g in gs:
    # 确保输入图是符合规范的
    if not is_valid_NAS201(g):
      result.append(None)
      continue

    # 解码图为矩阵
    m = decode_igraph_to_NAS201_matrix(g)

    # 通过格式化字符串来生成一个符合 NAS-Bench-201 规范的字符串
    types = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    result.append('|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(
      types[int(m[1][0])],
      types[int(m[2][0])], types[int(m[2][1])],
      types[int(m[3][0])], types[int(m[3][1])], types[int(m[3][2])]))

  # 返回所有图的解码结果列表
  return result


# 将图 g 解码为 NAS-Bench-201 的字符串表示
def decode_igraph_to_NAS_BENCH_201_string1(g):
  # 确保输入图是符合规范的
  if not is_valid_NAS201(g):
    return None
  # 解码图为矩阵
  m = decode_igraph_to_NAS201_matrix(g)
  # 通过格式化字符串来生成一个符合 NAS-Bench-201 规范的字符串
  types = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
  return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(
             types[int(m[1][0])],
             types[int(m[2][0])], types[int(m[2][1])],
             types[int(m[3][0])], types[int(m[3][1])], types[int(m[3][2])])


class Accumulator():
  def __init__(self, *args):
    self.args = args
    self.argdict = {}
    for i, arg in enumerate(args):
      self.argdict[arg] = i
    self.sums = [0] * len(args)
    self.cnt = 0

  def accum(self, val):
    val = [val] if type(val) is not list else val
    val = [v for v in val if v is not None]
    assert (len(val) == len(self.args))
    for i in range(len(val)):
      if torch.is_tensor(val[i]):
        val[i] = val[i].item()
      self.sums[i] += val[i]
    self.cnt += 1

  def clear(self):
    self.sums = [0] * len(self.args)
    self.cnt = 0

  def get(self, arg, avg=True):
    i = self.argdict.get(arg, -1)
    assert (i is not -1)
    if avg:
      return self.sums[i] / (self.cnt + 1e-8)
    else:
      return self.sums[i]

  def print_(self, header=None, time=None,
             logfile=None, do_not_print=[], as_int=[],
             avg=True):
    msg = '' if header is None else header + ': '
    if time is not None:
      msg += ('(%.3f secs), ' % time)

    args = [arg for arg in self.args if arg not in do_not_print]
    arg = []
    for arg in args:
      val = self.sums[self.argdict[arg]]
      if avg:
        val /= (self.cnt + 1e-8)
      if arg in as_int:
        msg += ('%s %d, ' % (arg, int(val)))
      else:
        msg += ('%s %.4f, ' % (arg, val))
    print(msg)

    if logfile is not None:
      logfile.write(msg + '\n')
      logfile.flush()

  def add_scalars(self, summary, header=None, tag_scalar=None, 
                    step=None, avg=True, args=None):
    for arg in self.args:
      val = self.sums[self.argdict[arg]]
      if avg:
        val /= (self.cnt + 1e-8)
      else:
        val = val
      tag = f'{header}/{arg}' if header is not None else arg
      if tag_scalar is not None:
        summary.add_scalars(main_tag=tag,
                            tag_scalar_dict={tag_scalar: val},
                            global_step=step)
      else:
        summary.add_scalar(tag=tag,
                           scalar_value=val,
                           global_step=step)


class Log:
  def __init__(self, args, logf, summary=None):
    self.args = args
    self.logf = logf
    self.summary = summary
    self.stime = time.time()
    self.ep_sttime = None

  def print(self, logger, epoch, tag=None, avg=True):
    if tag == 'train':
      ct = time.time() - self.ep_sttime
      tt = time.time() - self.stime
      msg = f'[total {tt:6.2f}s (ep {ct:6.2f}s)] epoch {epoch:3d}'
      print(msg)
      self.logf.write(msg+'\n')
    logger.print_(header=tag, logfile=self.logf, avg=avg)

    if self.summary is not None:
      logger.add_scalars(
        self.summary, header=tag, step=epoch, avg=avg)
    logger.clear()

  def print_args(self):
    argdict = vars(self.args)
    print(argdict)
    for k, v in argdict.items():
      self.logf.write(k + ': ' + str(v) + '\n')
    self.logf.write('\n')

  def set_time(self):
    self.stime = time.time()

  def save_time_log(self):
    ct = time.time() - self.stime
    msg = f'({ct:6.2f}s) meta-training phase done'
    print(msg)
    self.logf.write(msg+'\n')

  def print_pred_log(self, loss, corr, tau, tag, epoch=None, max_corr_dict=None):
    if tag == 'train':
      ct = time.time() - self.ep_sttime
      tt = time.time() - self.stime
      msg = f'[total {tt:6.2f}s (ep {ct:6.2f}s)] epoch {epoch:3d}'
      self.logf.write(msg+'\n'); print(msg); self.logf.flush() 
    #msg = f'ep {epoch:3d} ep time {time.time() - ep_sttime:8.2f} '
    #msg += f'time {time.time() - sttime:6.2f} '
    if max_corr_dict is not None:
      max_corr = max_corr_dict['corr']
      max_loss = max_corr_dict['loss']
      max_tau = max_corr_dict['tau']
      msg = f'{tag}: loss {loss:.6f} ({max_loss:.6f}) '
      msg += f'corr {corr:.4f} ({max_corr:.4f}) '
      msg += f'tau {tau:.4f} ({max_tau:.4f})'
    else:
      msg = f'{tag}: loss {loss:.6f} corr {corr:.4f} tau {tau:.4f}'
    self.logf.write(msg+'\n'); print(msg); self.logf.flush()   

  def max_corr_log(self, max_corr_dict):
    corr = max_corr_dict['corr']
    loss = max_corr_dict['loss']
    tau = max_corr_dict['tau']
    epoch = max_corr_dict['epoch']
    msg = f'[epoch {epoch}] max correlation: {corr:.4f}, max tau: {tau:.4f} loss: {loss:.6f}'
    self.logf.write(msg+'\n'); print(msg); self.logf.flush()


def get_log(epoch, loss, y_pred, y, acc_std, acc_mean, tag='train', acc_norm = True):
  # 归一化
  if(acc_norm):
    msg = f'[{tag}] Ep {epoch} loss {loss.item() / len(y):0.4f} '
    msg += f'pacc {y_pred[0]:0.4f}'
    msg += f'({y_pred[0] * 100.0 * acc_std + acc_mean:0.4f}) '
    msg += f'acc {y[0]:0.4f}({y[0] * 100 * acc_std + acc_mean:0.4f})'
  else:
    msg = f'[{tag}] Ep {epoch} loss {loss.item() / len(y):0.4f} '
    msg += f'pacc {y_pred[0]:0.4f} '
    msg += f'acc {y[0]:0.4f}'
  return msg

'''
def load_model(model, model_path, load_epoch=None, load_max_pt=None):
  if load_max_pt is not None:
    ckpt_path = os.path.join(model_path, load_max_pt)
  else:
    ckpt_path = os.path.join(model_path, f'ckpt_{load_epoch}.pt')
  print(f"==> load model from {ckpt_path} ...")
  model.cpu()
  model.load_state_dict(torch.load(ckpt_path))
'''
def load_model(model, model_path, load_epoch=None, load_max_pt=None):
  if load_max_pt is not None:
    ckpt_path = os.path.join(model_path, load_max_pt)
  else:
    ckpt_path = os.path.join(model_path, f'ckpt_{load_epoch}.pt')

  print(f"==> load model from {ckpt_path} ...")
  device = next(model.parameters()).device
  model.load_state_dict(torch.load(ckpt_path, map_location=device))
  model.to(device)


def save_model(epoch, model, model_path, max_corr=None):
  print("==> save current model...")
  if max_corr is not None:
    torch.save(model.cpu().state_dict(), 
                os.path.join(model_path, 'ckpt_max_corr.pt'))   
  else:
    torch.save(model.cpu().state_dict(), 
                os.path.join(model_path, f'ckpt_{epoch}.pt'))    



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h