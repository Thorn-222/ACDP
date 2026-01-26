from __future__ import print_function
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


def get_meta_train_loader(batch_size, data_path):
  dataset = MetaTrainDatabase(data_path)
  print(f'==> The number of tasks for meta-training: {len(dataset)}')

  loader = DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=1,
                      collate_fn=collate_fn)
  return loader


def get_meta_test_loader(data_path, data_name, num_class=None):
  if(data_name == 'cifar10' or data_name == 'cifar100'):
      dataset = MetaTestDataset_cifar(data_path, data_name, num_class)
      print(f'==> Meta-Test dataset {data_name}')

      loader = DataLoader(dataset=dataset,
                          batch_size=100,
                          shuffle=False,
                          num_workers=1,
                          collate_fn=collate_fn)
  else:
      dataset = MetaTestDataset(data_path, data_name, num_class)
      print(f'==> Meta-Test dataset {data_name}')

      loader = DataLoader(dataset=dataset,
                          batch_size=100,
                          shuffle=False,
                          num_workers=1,
                          collate_fn=collate_fn_data)

  return loader

class MetaTrainDatabase(Dataset):
  def __init__(self, data_path):
    self.mode = 'train'
    self.acc_norm = True

    mtr_data_path = os.path.join(data_path, 'imagenet_data.pt')

    data = torch.load(mtr_data_path)
    self.acc = data['acc']
    self.arch = data['arch']
    self.epoch_acc = data['epoch_acc']
    self.hj = data['hj']
    self.hj = torch.tensor(self.hj).float()
    self.acc = torch.tensor(self.acc)
    self.epoch_acc = torch.tensor(self.epoch_acc)

    idx_path = os.path.join(data_path, 'meta_train_tasks_idx.pt')
    random_idx_lst = torch.load(idx_path)
    self.idx_lst = {}
    self.idx_lst['valid'] = random_idx_lst[:40]
    self.idx_lst['train'] = random_idx_lst[40:]

    self.mean = torch.mean(self.acc[self.idx_lst['train']]).item()
    self.std = torch.std(self.acc[self.idx_lst['train']]).item()
    train_acc = self.acc[self.idx_lst['train']]
    if self.acc_norm:
        train_acc = ((train_acc - self.mean) / self.std) / 100.0
    else:
        train_acc = train_acc / 100.0
    self.threshold = train_acc.median().item()

  def set_mode(self, mode):
    self.mode = mode

  def get_threshold(self):
    return self.threshold

  def __len__(self):
    return len(self.idx_lst[self.mode])

  def __getitem__(self, index):
    ridx = self.idx_lst[self.mode]
    arch = self.arch[ridx[index]]
    acc = self.acc[ridx[index]]

    epoch_acc = self.epoch_acc[ridx[index]]
    hj = self.hj[ridx[index]]

    # 归一化准确率
    if self.acc_norm:
      acc = ((acc- self.mean) / self.std) / 100.0
    else:
      acc = acc / 100.0


    return epoch_acc, hj, arch, acc



class MetaTestDataset_cifar(Dataset):
  def __init__(self, data_path, data_name, num_class=None):
    self.data_name = data_name
    self.acc_norm = True

    num_class_dict = {
        'cifar100': 100,
        'cifar10': 10
    }
    if num_class is not None:
      self.num_class = num_class
    else:
      self.num_class = num_class_dict[data_name]

    self.acc = torch.load(os.path.join(data_path, 'nasbench201.pt'))['test-acc'][data_name]
    self.acc = torch.tensor(self.acc, dtype=torch.float32)
    self.mean = torch.mean(self.acc).item()
    self.std = torch.std(self.acc).item()

    mtr_data_path = os.path.join(data_path, f'{data_name}_data.pt')
    data = torch.load(mtr_data_path)
    self.epoch_acc = data['epoch_acc']
    self.epoch_acc = torch.tensor(self.epoch_acc, dtype=torch.float32)
    self.arch = data['arch']
    self.hj = data['hj']
    self.hj = torch.tensor(self.hj, dtype=torch.float32)

    '''
        # -------------------------------------------------------
        # Hierarchical Evaluation Strategy：Sort by hj and keep the top 10%(20%).
        # -------------------------------------------------------
        sorted_indices = torch.argsort(self.zs, descending=True)

        keep_count = int(len(sorted_indices) * 0.10)
        top_indices = sorted_indices[:keep_count]

        top_indices = torch.sort(top_indices)[0]
        top_indices_list = top_indices.tolist()

        print(f"==> {data_name} Pre-filter (preserving original order): Keep the first 10% ({keep_count} items)")

        self.hj = self.hj[top_indices]
        self.feature = [self.feature[i] for i in top_indices_list]
        self.epoch_acc = self.epoch_acc[top_indices]
        self.arch = [self.arch[i] for i in top_indices_list]
        # -------------------------------------------------------
        self.acc = torch.tensor(0, dtype=torch.float32)
        self.mean = torch.mean(self.acc).item()
        self.std = torch.std(self.acc).item()
    '''

  def __len__(self):
    return len(self.arch)

  def __getitem__(self, index):
    arch = self.arch[index]
    acc = self.acc[index]
    epoch_acc = self.epoch_acc[index]
    hj = self.hj[index]
    if self.acc_norm:
      acc = ((acc - self.mean) / self.std) / 100.0
    else:
      acc = acc / 100.0

    return  epoch_acc, hj, arch, acc


def collate_fn(batch):
    e_acc = torch.stack([item[0] for item in batch])
    hj = torch.stack([item[1] for item in batch])
    arch = [item[2] for item in batch]
    acc = torch.stack([item[3] for item in batch])

    return [e_acc, hj, arch, acc]


class MetaTestDataset(Dataset):
    def __init__(self, data_path, data_name, num_class=None):
        self.data_name = data_name
        self.acc_norm = True

        num_class_dict = {
            'mnist': 10,
            'svhn': 10,
            'aircraft': 30,
            'pets': 37
        }
        if num_class is not None:
            self.num_class = num_class
        else:
            self.num_class = num_class_dict[data_name]

        mtr_data_path = os.path.join(data_path, f'{data_name}_data.pt')
        data = torch.load(mtr_data_path)
        self.epoch_acc = data['epoch_acc']
        self.epoch_acc = torch.tensor(self.epoch_acc, dtype=torch.float32)
        self.arch = data['arch']
        self.hj = data['hj']
        self.hj = torch.tensor(self.hj, dtype=torch.float32)

        '''
            # -------------------------------------------------------
            # Hierarchical Evaluation Strategy：Sort by hj and keep the top 10%(20%).
            # -------------------------------------------------------
            sorted_indices = torch.argsort(self.zs, descending=True)
    
            keep_count = int(len(sorted_indices) * 0.10)
            top_indices = sorted_indices[:keep_count]
    
            top_indices = torch.sort(top_indices)[0]
            top_indices_list = top_indices.tolist()
    
            print(f"==> {data_name} Pre-filter (preserving original order): Keep the first 10% ({keep_count} items)")
    
            self.hj = self.hj[top_indices]
            self.feature = [self.feature[i] for i in top_indices_list]
            self.epoch_acc = self.epoch_acc[top_indices]
            self.arch = [self.arch[i] for i in top_indices_list]
            # -------------------------------------------------------
        '''
        # placeholder
        self.acc = torch.tensor(0, dtype=torch.float32)
        self.mean = torch.mean(self.acc).item()
        self.std = torch.std(self.acc).item()

    def __len__(self):
        return len(self.arch)

    def __getitem__(self, index):
        arch = self.arch[index]
        epoch_acc = self.epoch_acc[index]
        hj = self.hj[index]

        return epoch_acc, hj, arch

def collate_fn_data(batch):
    e_acc = torch.stack([item[0] for item in batch])
    hj = torch.stack([item[1] for item in batch])
    arch = [item[2] for item in batch]

    return [e_acc, hj, arch]