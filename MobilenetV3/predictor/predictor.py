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

from loader import get_meta_train_loader
from predictor.predictor_model import EnhancedPredictorWithHyperNet

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

    self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
    self.mtrloader = get_meta_train_loader(self.batch_size, self.data_path, self.num_sample, is_pred=True)

    self.acc_mean = self.mtrloader.dataset.mean
    self.acc_std = self.mtrloader.dataset.std

    self.mtrlog = Log(self.args, open(os.path.join(self.save_path, self.model_name, 'meta_train_predictor.log'), 'w'))
    self.mtrlog.print_args()

  def forward(self, arch, d):
    if isinstance(d, (float, int, np.float64)):
      d_tensor = torch.tensor([[d]], dtype=torch.float32).to(self.device)
    else:
      d_tensor = d.view(-1, 1).to(self.device)

    if isinstance(arch, dict):
      arch_batch = [arch]
    else:
      arch_batch = arch

    num, adj, ope = self.model.get_arch(arch_batch)
    num, adj, ope = num.to(self.device), adj.to(self.device), ope.to(self.device)
    pred_score = self.model(num, adj, ope, d_tensor)

    return pred_score


  def meta_train(self):
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

    for arch, d, acc in pbar:
      self.optimizer.zero_grad()
      y_pred = self.forward(arch, d)
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

    with torch.no_grad():
      for arch, d, acc in pbar:
        y_pred = self.forward(arch,d)
        y = acc.to(self.device)

        loss = self.model.loss(y_pred, y)

        y = y.tolist()
        y_pred = y_pred.squeeze().tolist()
        y_all += y
        y_pred_all += y_pred
        pbar.set_description(get_log(epoch, loss, y_pred, y, self.acc_std, self.acc_mean, tag='val'))
        valoss += float(loss)

    return valoss/dlen, pearsonr(np.array(y_all), np.array(y_pred_all))[0], kendalltau(np.array(y_all*10), np.array(y_pred_all*10))[0]



