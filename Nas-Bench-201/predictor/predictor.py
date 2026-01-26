from __future__ import print_function
import torch
import os
import random

from numpy import shape
from tqdm import tqdm
import numpy as np
import time
import os
import shutil

from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr, kendalltau

from utils import load_graph_config, decode_igraph_to_NAS_BENCH_201_string
from utils import Log, get_log
from utils import load_model, save_model, mean_confidence_interval

from loader import get_meta_train_loader, get_meta_test_loader, MetaTestDataset
from loader import get_meta_test_loader
from predictor.predictor_model import EnhancedPredictorWithHyperNet
from nas_bench_201 import train_single_model

class Predictor:
  def __init__(self, args):
    self.args = args
    self.batch_size = args.batch_size
    self.data_path = args.data_path
    self.num_sample = args.num_sample
    self.max_epoch = args.max_epoch
    self.save_epoch = args.save_epoch
    self.model_path = args.model_path
    self.save_path = args.save_path
    self.model_name = args.model_name
    self.data_name = args.data_name
    self.test = args.test
    self.device = torch.device("cuda:0")
    self.max_corr_dict = {'corr': -1, 'epoch': -1, 'loss':-1, 'tau':-1}
    self.train_arch = args.train_arch


    self.model = EnhancedPredictorWithHyperNet()
    self.model.device = self.device
    self.model.to(self.device)


    if self.test:
      self.data_name = args.data_name
      self.num_class = args.num_class
      self.load_epoch = args.load_epoch
      load_model(self.model, self.model_path, load_max_pt='ckpt_max_corr.pt')
      self.mtrloader = get_meta_test_loader(self.data_path, self.data_name)
      self.acc_mean = self.mtrloader.dataset.mean
      self.acc_std = self.mtrloader.dataset.std

      self.mtrlog = Log(self.args, open(os.path.join(self.save_path, self.model_name, 'meta_test_predictor.log'), 'w'))
      self.mtrlog.print_args()

    else:
      self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
      self.mtrloader =  get_meta_train_loader(self.batch_size, self.data_path)

      self.acc_mean = self.mtrloader.dataset.mean
      self.acc_std = self.mtrloader.dataset.std

      self.mtrlog = Log(self.args, open(os.path.join(self.save_path, self.model_name, 'meta_train_predictor.log'), 'w'))
      self.mtrlog.print_args()

  def forward(self, e_acc, hj, arch):
    e_acc = e_acc.unsqueeze(1).to(self.device)
    hj = hj.unsqueeze(1).to(self.device)
    # Hybrid-Fidelity Dataset Trait
    d = e_acc+hj
    # Arch Feature
    num, adj, ope = self.model.get_arch(arch)
    y_pred = self.model(num, adj, ope,d)

    return y_pred

  def meta_train(self):
    sttime = time.time()
    for epoch in range(1, self.max_epoch + 1):
      self.mtrlog.ep_sttime = time.time()

      loss, corr, tau = self.meta_train_epoch(epoch)
      self.scheduler.step(tau)
      self.mtrlog.print_pred_log(loss, corr, tau, 'train', epoch)

      valoss, vacorr, vatau = self.meta_validation(epoch)
      if self.max_corr_dict['tau'] < vatau:
        self.max_corr_dict['tau'] = vatau
        self.max_corr_dict['corr'] = vacorr
        self.max_corr_dict['epoch'] = epoch
        self.max_corr_dict['loss'] = valoss
        save_model(epoch, self.model, self.model_path, max_corr=True)
      self.mtrlog.print_pred_log(valoss, vacorr, vatau, 'valid', max_corr_dict=self.max_corr_dict)

      if epoch % self.save_epoch == 0:
        save_model(epoch, self.model, self.model_path)

    self.mtrlog.save_time_log()
    self.mtrlog.max_corr_log(self.max_corr_dict)

  def meta_train_epoch(self, epoch):
    self.model.to(self.device)
    self.model.train()
    self.mtrloader.dataset.set_mode('train')

    dlen = len(self.mtrloader.dataset)
    trloss = 0
    y_all, y_pred_all, y_epoch_all =[], [], []
    pbar = tqdm(self.mtrloader, ncols=200)

    for e_acc, hj, arch, acc in pbar:
      self.optimizer.zero_grad()
      y_pred = self.forward(e_acc, hj, arch)
      y = acc.to(self.device)
      loss = self.model.loss(y_pred, y)
      loss.backward()
      self.optimizer.step()
      y = y.tolist()
      y_pred = y_pred.squeeze().tolist()
      y_all += y
      y_pred_all += y_pred
      pbar.set_description(get_log(epoch, loss, y_pred, y, self.acc_std, self.acc_mean))
      trloss += float(loss)

    return trloss/dlen, pearsonr(np.array(y_all), np.array(y_pred_all))[0], kendalltau(np.array(y_all*10), np.array(y_pred_all*10))[0]

  def meta_validation(self, epoch):
    self.model.to(self.device)
    self.model.eval()

    valoss = 0
    self.mtrloader.dataset.set_mode('valid')
    dlen = len(self.mtrloader.dataset)
    y_all, y_pred_all,  y_epoch_all=[], [], []
    pbar = tqdm(self.mtrloader, ncols=200)

    # 禁止梯度计算
    with torch.no_grad():
      for e_acc, hj, arch, acc in pbar:
        y_pred = self.forward(e_acc, hj, arch)
        y = acc.to(self.device)

        loss = self.model.loss(y_pred, y)
        y = y.tolist()
        y_pred = y_pred.squeeze().tolist()
        y_all += y
        y_pred_all += y_pred
        pbar.set_description(get_log(epoch, loss, y_pred, y, self.acc_std, self.acc_mean, tag='val'))
        valoss += float(loss)

    return valoss/dlen, pearsonr(np.array(y_all), np.array(y_pred_all))[0], kendalltau(np.array(y_all*10), np.array(y_pred_all*10))[0]

  def meta_test_cifar(self):
    for epoch in range(1, 2):
      self.mtrlog.ep_sttime = time.time()

      valoss, vacorr, vatau = self.meta_test_epoch(epoch)
      if self.max_corr_dict['tau'] < vatau:
        self.max_corr_dict['tau'] = vatau
        self.max_corr_dict['corr'] = vacorr
        self.max_corr_dict['epoch'] = epoch
        self.max_corr_dict['loss'] = valoss
      self.mtrlog.print_pred_log(valoss, vacorr, vatau, 'test', max_corr_dict=self.max_corr_dict)

    self.mtrlog.save_time_log()
    self.mtrlog.max_corr_log(self.max_corr_dict)

  def meta_test_epoch(self, epoch):
    self.model.to(self.device)
    self.model.eval()

    valoss = 0
    dlen = len(self.mtrloader.dataset)
    y_all, y_pred_all = [], []
    pbar = tqdm(self.mtrloader, ncols=200)

    with torch.no_grad():
      for e_acc, hj, arch, acc in pbar:
        y_pred = self.forward(e_acc, hj, arch)
        y = acc.to(self.device)

        loss = self.model.loss(y_pred, y)
        y = y.tolist()
        y_pred = y_pred.squeeze().tolist()
        y_all += y
        y_pred_all += y_pred
        pbar.set_description(get_log(epoch, loss, y_pred, y, self.acc_std, self.acc_mean, tag='val'))
        valoss += float(loss)

    return valoss / dlen, pearsonr(np.array(y_all), np.array(y_pred_all))[0], kendalltau(np.array(y_all * 10), np.array(y_pred_all * 10))[0]

  def meta_test(self):
    if self.data_name == 'cifar10' or self.data_name == 'cifar100':
     self.meta_test_cifar()
    else:
      self.meta_test_per_dataset(self.data_name)


  def load_generated_archs(self):
    mtest_path = os.path.join(self.data_path, 'nasbench201_arch.pt')
    gen_arch_str = torch.load(mtest_path)
    return gen_arch_str

  def get_items(self, full_target, full_source, source):
    return [full_target[full_source.index(_)] for _ in source]

  def collect_data(self, arch_igraph):
    x_batch, g_batch = [], []
    for _ in range(10):
      x_batch.append(self.test_dataset[0])
      g_batch.append(arch_igraph)
    return torch.stack(x_batch).to(self.device), g_batch

  def select_top_arch(self, y_pred_all, gen_arch_str, N):
    _, sorted_idx = torch.sort(y_pred_all, descending=True)
    sotred_gen_arch_str = [gen_arch_str[_] for _ in sorted_idx]
    final_str = sotred_gen_arch_str[:N]
    return final_str

  def select_top_acc(self, data_name, final_str):
    final_test_acc = self.get_items(
      full_target=self.nasbench201['test-acc'][data_name],
      full_source=self.nasbench201['arch']['str'],
      source=final_str)
    max_test_acc = max(final_test_acc)
    return max_test_acc

  def meta_test_per_dataset(self, data_name):
    self.nasbench201 = torch.load(os.path.join(self.data_path, 'nasbench201.pt'))
    self.test_dataset = MetaTestDataset(self.data_path, data_name, self.num_class)

    meta_test_path = os.path.join(self.save_path, 'meta_test', data_name, 'best_arch')
    if not os.path.exists(meta_test_path):
      os.makedirs(meta_test_path)

    f_arch_str = open(os.path.join(meta_test_path, 'architecture.txt'), 'w')
    save_path = os.path.join(meta_test_path, 'accuracy.txt')
    f = open(save_path, 'w')

    arch_runs = []
    elasped_time = []
    N = 1
    runs = 1

    print(f'==> select top architectures for {data_name} by meta-predictor...')
    for run in range(1, runs + 1):
      print(f'==> run #{run}')
      gen_arch_str = self.load_generated_archs()

      y_pred_all = []
      #gen_arch_str = []
      self.model.eval()
      self.model.to(self.device)
      pbar = tqdm(self.mtrloader, ncols=200)

      sttime = time.time()
      with torch.no_grad():
        for e_acc, hj, arch in pbar:
          y_pred = self.forward(e_acc, hj, arch)
          y_pred = y_pred.squeeze()
          y_pred_all.append(y_pred.cpu().detach())

      all_y_pred_tensor = torch.cat(y_pred_all)

      top_arch_lst = self.select_top_arch(all_y_pred_tensor, gen_arch_str, N)
      arch_runs.append(top_arch_lst[0])

      elasped = time.time() - sttime
      elasped_time.append(elasped)

      if 'cifar' in data_name:
        acc = self.select_top_acc(data_name, top_arch_lst)
        acc_runs.append(acc)

    for run, arch_str in enumerate(arch_runs):
      f_arch_str.write(f'{arch_str}\n'); print(f'{arch_str}')

    time_path = os.path.join(self.save_path, 'meta_test', data_name, 'time.txt')
    with open(time_path, 'a') as f_time:
      msg = f'predictor average elasped time {np.mean(elasped_time):.2f}s'
      print(f'==> save time in {time_path}')
      f_time.write(msg+'\n'); print(msg)

    if self.train_arch:
      if not 'cifar' in data_name:
        acc_runs = self.train_single_arch(data_name, arch_runs[0], meta_test_path)
      print(f'==> save results in {save_path}')
      for r, acc in enumerate(acc_runs):
        msg = f'run {r+1} {acc:.2f} (%)'
        f.write(msg+'\n'); print(msg)

      m, h = mean_confidence_interval(acc_runs)
      msg = f'Avg {m:.2f}+-{h.item():.2f} (%)'
      f.write(msg+'\n'); print(msg)

  def train_single_arch(self, data_name, arch_str, meta_test_path):
    # Use the same random seed to ensure the experiment is reproducible.
    if (data_name == 'mnist'):
      seeds = (111,148,555)
    elif (data_name == 'svhn'):
      seeds = (927,877,2677)
    elif (data_name == 'aircraft'):
      seeds = (731,444,108)
    elif (data_name == 'pets'):
      seeds = (937,2983,9811)
    train_single_model(save_dir=meta_test_path,
                       workers=8,
                       datasets=[data_name],
                       xpaths=[f'{self.data_path}/raw-data/{data_name}'],
                       splits=[0],
                       use_less=False,
                       seeds=seeds,
                       model_str=arch_str,
                       arch_config={'channel': 16, 'num_cells': 5})

    test_acc_lst = []
    for seed in seeds:
      result = torch.load(os.path.join(meta_test_path, f'seed-{seed:04d}.pth'))
      valid_acc_history = result[data_name]['valid_acc1es']
      all_test_accuracies = [
        acc_value
        for key, acc_value in valid_acc_history.items()
        if key.startswith('x-test@')
      ]

      if all_test_accuracies:
        max_acc = max(all_test_accuracies)
        test_acc_lst.append(max_acc)
      else:
        print(f"Warning: Could not find 'x-test@' accuracies for seed {seed} in {data_name} results.")

    return test_acc_lst

  def select_top_arch_acc(self, data_name, y_pred_all, gen_arch_str, N):
    _, sorted_idx = torch.sort(y_pred_all, descending=True)
    gen_test_acc = self.get_items(
              full_target=self.nasbench201['test-acc'][data_name], 
              full_source=self.nasbench201['arch']['str'], 
              source=gen_arch_str)
    sorted_gen_test_acc = torch.tensor(gen_test_acc)[sorted_idx]
    sotred_gen_arch_str = [gen_arch_str[_] for _ in sorted_idx]    
    
    max_idx = torch.argmax(sorted_gen_test_acc[:N]).item()
    final_acc = sorted_gen_test_acc[:N][max_idx]
    final_str = sotred_gen_arch_str[:N][max_idx]
    return final_acc, final_str



