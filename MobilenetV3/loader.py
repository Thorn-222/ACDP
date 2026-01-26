from __future__ import print_function
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


def get_meta_train_loader(batch_size, data_path, num_sample, is_pred=False):
  dataset = MetaTrainDatabase(data_path, num_sample, is_pred)
  print(f'==> The number of tasks for meta-training: {len(dataset)}')

  loader = DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=1,
                      collate_fn=collate_fn)
  return loader


class MetaTrainDatabase(Dataset):
  def __init__(self, data_path, num_sample, is_pred=False):
    self.mode = 'train'
    self.acc_norm = True
    self.num_sample = num_sample

    mv3_data_path = os.path.join(data_path, 'imagenet_data.pt')
    data = torch.load(mv3_data_path)
    self.acc = data['acc']
    self.arch = data['arch']
    self.D = data['D']

    random_idx_lst = list(range(len(self.D)))
    self.idx_lst = {}
    self.idx_lst['valid'] = random_idx_lst[:40]
    self.idx_lst['train'] = random_idx_lst[40:]
    self.acc = torch.tensor(self.acc)
    self.mean = torch.mean(self.acc[self.idx_lst['train']]).item()
    self.std = torch.std(self.acc[self.idx_lst['train']]).item()

  def set_mode(self, mode):
    self.mode = mode

  def get_threshold(self):
    return self.threshold

  def __len__(self):
    return len(self.idx_lst[self.mode])

  def __getitem__(self, index):
    acc = self.acc[index]
    D = self.D[index]
    arch = self.arch[index]

    if self.acc_norm:
      acc = (acc- self.mean) / self.std
    else:
      acc = acc

    return arch, D, acc


def collate_fn(batch):
    arch = [item[0] for item in batch]
    d = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    acc = torch.tensor([item[2] for item in batch], dtype=torch.float32)

    return arch, d, acc