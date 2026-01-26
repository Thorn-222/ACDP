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
  device = torch.device("cuda:1")
  torch.cuda.manual_seed(args.seed)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)

  args.model_name = 'predictor'
  args.model_path = os.path.join(args.save_path, args.model_name, 'model')
  if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

  p = Predictor(args)
  if args.test:
    print("Since there is no known dataset for architecture performance data in the MobilenetV3 search space, testing is not supported. Please refer to the Evolutionary Search Algorithm ENAS.py instead.")
  else:
    p.meta_train()


if __name__ == '__main__':
  main()
