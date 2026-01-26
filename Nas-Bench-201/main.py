import os
import random
import numpy as np

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from parser import get_parser
from predictor.predictor import Predictor


def main():
  args = get_parser()
  os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
  torch.cuda.manual_seed(args.seed)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)
  args.model_path = os.path.join(args.save_path, args.model_name, 'model')
  if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)
      
  p = Predictor(args)
  if args.test:
    p.meta_test()
  else:
    p.meta_train()


if __name__ == '__main__':
  main()
