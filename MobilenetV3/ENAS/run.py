import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import re
import copy
import argparse
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

from transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.evaluator import OFAEvaluator
from transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.codebase.data_providers.aircraft import FGVCAircraft
from transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.codebase.data_providers.pets2 import PetDataset
from transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.codebase.data_providers.autoaugment import CIFAR10Policy

DATASET_CONFIGS = {
    'mnist': {
        'n_cls': 10, 'epochs': 50, 'is_gray': True,
        'mean': [0.1307], 'std': [0.3081],
        'data_path': '/data/MNIST'
    },
    'cifar10': {
        'n_cls': 10, 'epochs': 150, 'is_gray': False,
        'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616],
        'data_path': '/data/cifar10'
    },
    'cifar100': {
        'n_cls': 100, 'epochs': 150, 'is_gray': False,
        'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761],
        'data_path': '/data/cifar100'
    },
    'aircraft': {
        'n_cls': 100, 'epochs': 150, 'is_gray': False,
        'mean': [0.4893, 0.5184, 0.5388], 'std': [0.2239, 0.2164, 0.2462],
        'data_path': '/data/FGVC Aircraft'
    },
    'pets': {
        'n_cls': 37, 'epochs': 150, 'is_gray': False,
        'mean': [0.4829, 0.4448, 0.3957], 'std': [0.2593, 0.2533, 0.2598],
        'data_path': '/data/Oxford'
    }
}


def set_global_seed(seed=42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataloader(dataset_name, config):
    img_size = 224
    t_train = [transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC)]
    t_test = [transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC)]

    if config['is_gray']:
        t_train.append(transforms.Grayscale(num_output_channels=3))
        t_test.append(transforms.Grayscale(num_output_channels=3))
        mean, std = config['mean'] * 3, config['std'] * 3
    else:
        mean, std = config['mean'], config['std']

    if 'cifar' in dataset_name: t_train.append(CIFAR10Policy())
    t_train += [transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                transforms.RandomErasing(p=1, scale=(0.02, 0.02), ratio=(1, 1), value=0),
                transforms.Normalize(mean, std)]
    t_test += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    train_transform = transforms.Compose(t_train)
    test_transform = transforms.Compose(t_test)

    if dataset_name == 'mnist':
        train_set = datasets.MNIST(config['data_path'], train=True, transform=train_transform, download=True)
        test_set = datasets.MNIST(config['data_path'], train=False, transform=test_transform, download=True)
    elif dataset_name == 'cifar10':
        train_set = datasets.CIFAR10(config['data_path'], train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR10(config['data_path'], train=False, transform=test_transform, download=True)
    elif dataset_name == 'cifar100':
        train_set = datasets.CIFAR100(config['data_path'], train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR100(config['data_path'], train=False, transform=test_transform, download=True)
    elif dataset_name == 'aircraft':
        train_set = FGVCAircraft(config['data_path'], class_type='variant', split='trainval', transform=train_transform)
        test_set = FGVCAircraft(config['data_path'], class_type='variant', split='test', transform=test_transform)
    elif dataset_name == 'pets':
        train_set = PetDataset(config['data_path'], train=True, num_cl=config['n_cls'], val_split=0.15,transforms=train_transform)
        test_set = PetDataset(config['data_path'], train=False, num_cl=config['n_cls'], val_split=0.15,transforms=test_transform)

    return DataLoader(train_set, batch_size=96, shuffle=True, num_workers=8), DataLoader(test_set, batch_size=200, shuffle=False, num_workers=8)


def set_architecture(arch_str, evaluator, n_cls, device):
    arch_config = json.loads(arch_str.replace("'", '"'))
    evaluator.engine.set_active_subnet(ks=arch_config['ks'], e=arch_config['e'], d=arch_config['d'])
    net = evaluator.engine.get_active_subnet()

    if hasattr(net, 'classifier'):
        net.classifier.linear = nn.Linear(net.classifier.linear.in_features, n_cls)
    else:
        evaluator.engine.classifier.linear = nn.Linear(evaluator.engine.classifier.linear.in_features, n_cls)
        net = evaluator.engine.get_active_subnet()

    return net.to(device)


def train_one(arch_str, dataset_name, evaluator, device):
    config = DATASET_CONFIGS[dataset_name]
    train_loader, test_loader = get_dataloader(dataset_name, config)
    net = set_architecture(arch_str, evaluator, config['n_cls'], device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9, weight_decay=4e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])

    best_acc = 0.0
    for epoch in range(config['epochs']):
        net.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(net(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
        scheduler.step()

        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
        acc = 100. * correct / total
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='mnist',choices=['mnist', 'cifar10', 'cifar100', 'aircraft', 'pets'])
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OFA_PATH = '/MobilenetV3/data/ofa_mbv3_d234_e346_k357_w1.0'
    evaluator = OFAEvaluator(num_gen_arch=None, img_size=224, drop_path=0.2, model_path=OFA_PATH)

    base_dir = f'/MobilenetV3/ENAS/{args.data_name}/'
    log_path = os.path.join(base_dir, f'search_trajectory_parents_{args.data_name}.txt')
    res_json = os.path.join(base_dir, f'official_training_results_{args.data_name}.json')

    tasks = []
    pattern = re.compile(r"Gen (\d+) \| .* Arch: (\{.*\})")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match: tasks.append({"gen": int(match.group(1)), "arch": match.group(2)})

    history = json.load(open(res_json, 'r')) if os.path.exists(res_json) else {}

    for task in tasks:
        if task['arch'] in history:
            continue
        print(f"\n>>> 训练 {args.data_name.upper()} Gen {task['gen']}...")
        acc = train_one(task['arch'], args.data_name, evaluator, device)
        history[task['arch']] = acc
        json.dump(history, open(res_json, 'w'), indent=4)

    output_txt_path = os.path.join(base_dir, f'acc_official_{args.data_name}.txt')

    with open(output_txt_path, 'w') as f:
        for task in tasks:
            acc = history.get(task['arch'], "N/A")
            if isinstance(acc, (int, float)):
                f.write(f"Gen {task['gen']}: {acc:.2f}%\n")
            else:
                f.write(f"Gen {task['gen']}: {acc}\n")

    print(f"All assessment tasks are completed and summary files have been generated.")