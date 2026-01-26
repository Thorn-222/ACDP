import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image

from ENAS.transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.evaluator import OFAEvaluator
from ENAS.transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.codebase.data_providers.aircraft import FGVCAircraft
from ENAS.transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.codebase.data_providers.autoaugment import CIFAR10Policy
from predictor.predictor import Predictor
from parser import get_parser
from utils import load_model

DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10,
        'mean': [0.49139968, 0.48215827, 0.44653124],
        'std': [0.24703233, 0.24348505, 0.26158768],
        'batch_limit': 100,
        'subdir': 'cifar10_extracted'
    },
    'cifar100': {
        'num_classes': 100,
        'mean': [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
        'std': [0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
        'batch_limit': 100,
        'subdir': 'cifar100_extracted'
    },
    'mnist': {
        'num_classes': 10,
        'mean': [0.1307, 0.1307, 0.1307],
        'std': [0.3081, 0.3081, 0.3081],
        'batch_limit': 50,
        'subdir': 'MNIST'
    },
    'svhn': {
        'num_classes': 10,
        'mean': [0.4376821, 0.4437697, 0.4728091],
        'std': [0.19803012, 0.20101562, 0.19703614],
        'batch_limit': 80,
        'subdir': 'SVHN'
    },
    'pets': {
        'num_classes': 37,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'batch_limit': None,
        'subdir': 'Oxford'
    },
    'aircraft': {
        'num_classes': 100,
        'mean': [0.48933587, 0.51835374, 0.53879144],
        'std': [0.22388883, 0.21641635, 0.24615605],
        'batch_limit': None,
        'subdir': 'FGVC Aircraft'
    }
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class UnifiedEvolutionNAS:
    def __init__(self, dataset_name, base_path='/data'):
        self.cfg = DATASET_CONFIGS[dataset_name]
        self.dataset_name = dataset_name
        self.device = torch.device("cuda:0")
        self.img_size = 224
        self.ofa_path = '/MobilenetV3/data/ofa_mbv3_d234_e346_k357_w1.0'

        self.parent_history_file = f'search_trajectory_{dataset_name}.txt'
        self.cache_file = f'D_full_ofa_{dataset_name}.json'
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.D_vault = json.load(f)
            print(f"Number of loaded cache architectures: {len(self.D_vault)}")
        else:
            self.D_vault = {}

        self.evaluator = OFAEvaluator(num_gen_arch=None, img_size=self.img_size, drop_path=0.2,model_path=self.ofa_path)

        self._init_dataloaders(os.path.join(base_path, self.cfg['subdir']))

    def _init_dataloaders(self, data_path):
        t_train = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy() if self.dataset_name != 'mnist' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
            transforms.Normalize(self.cfg['mean'], self.cfg['std']),
        ])
        t_test = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(self.cfg['mean'], self.cfg['std']),
        ])

        if self.dataset_name == 'aircraft':
            train_set = FGVCAircraft(data_path, class_type='variant', split='trainval', transform=t_train)
            test_set = FGVCAircraft(data_path, class_type='variant', split='test', transform=t_test)
        elif self.dataset_name == 'svhn':
            train_set = datasets.SVHN(data_path, split='train', download=True, transform=t_train)
            test_set = datasets.SVHN(data_path, split='test', download=True, transform=t_test)
        elif self.dataset_name == 'pets':
            train_set = datasets.OxfordIIITPet(os.path.dirname(data_path), split='trainval', download=True,transform=t_train)
            test_set = datasets.OxfordIIITPet(os.path.dirname(data_path), split='test', download=True, transform=t_test)
        else:  # CIFARs, MNIST
            ds_cls = datasets.CIFAR10 if self.dataset_name == 'cifar10' else datasets.CIFAR100 if self.dataset_name == 'cifar100' else datasets.MNIST
            train_set = ds_cls(data_path, train=True, download=True, transform=t_train)
            test_set = ds_cls(data_path, train=False, download=True, transform=t_test)
            if self.dataset_name == 'mnist':
                train_set.transform = transforms.Compose([transforms.Grayscale(3), t_train])
                test_set.transform = transforms.Compose([transforms.Grayscale(3), t_test])

        self.train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

        hj_loader = DataLoader(train_set, batch_size=128, shuffle=False)
        self.hj_batch_data, _ = next(iter(hj_loader))
        self.hj_batch_data = self.hj_batch_data.to(self.device)

    def get_ofa_subnet(self, arch):
        self.evaluator.engine.set_active_subnet(ks=arch['ks'], e=arch['e'], d=arch['d'])
        net = self.evaluator.engine.get_active_subnet()
        if hasattr(net, 'classifier'):
            in_f = net.classifier.linear.in_features
            net.classifier.linear = nn.Linear(in_f, self.cfg['num_classes'])
        return net.to(self.device)

    def run_proxy_train(self, arch, epochs):
        net = self.get_ofa_subnet(arch)
        if epochs == 1:  # H-Jacobian
            net.eval()
            net.K = np.zeros((128, 128))

            def hook_fn(m, inp, out):
                if isinstance(inp, tuple): inp = inp[0]
                x = (inp.view(inp.size(0), -1) > 0).float()
                K = x @ x.t() + (1. - x) @ (1. - x.t())
                net.K += K.detach().cpu().numpy()

            hooks = [m.register_forward_hook(hook_fn) for m in net.modules() if
                     any(t in str(type(m)) for t in ['ReLU', 'Hswish', 'Hardswish'])]
            with torch.no_grad():
                net(self.hj_batch_data)
            for h in hooks: h.remove()
            _, score = np.linalg.slogdet(net.K + np.eye(128) * 1e-6)
            return float(score)
        else:  # 5-epoch Acc
            opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=4e-5)
            crit = nn.CrossEntropyLoss(label_smoothing=0.1)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
            for _ in range(5):
                net.train()
                for i, (img, lbl) in enumerate(self.train_loader):
                    if self.cfg['batch_limit'] and i >= self.cfg['batch_limit']: break
                    img, lbl = img.to(self.device), lbl.to(self.device)
                    opt.zero_grad()
                    crit(net(img), lbl).backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5)
                    opt.step()
                sch.step()
            net.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for img, lbl in self.test_loader:
                    img, lbl = img.to(self.device), lbl.to(self.device)
                    _, p = net(img).max(1)
                    total += lbl.size(0)
                    correct += p.eq(lbl).sum().item()
            return 100. * correct / total

    def get_metrics(self, arch):
        s = json.dumps(arch, sort_keys=True)
        if s not in self.D_vault:
            self.D_vault[s] = (self.run_proxy_train(arch, 1), self.run_proxy_train(arch, 5))
            with open(self.cache_file, 'w') as f: json.dump(self.D_vault, f)
        return self.D_vault[s]

    def run_search(self, predictor, pop_size=30, generations=100):
        # Initial population
        save_dir = '/MobilenetV3/data'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        init_pop_path = os.path.join(save_dir, f"initial_population_{self.dataset_name}.pt")

        if os.path.exists(init_pop_path):
            print(f" Found the pre-stored starting point, loading the initial population of {self.dataset_name}: {init_pop_path}")
            pop = torch.load(init_pop_path)
        else:
            print(f" No origin found, generating and saving initial population of {self.dataset_name}: {init_pop_path}")
            pop = [{
                'ks': [random.choice([3, 5, 7]) for _ in range(20)],
                'e': [random.choice([3, 4, 6]) for _ in range(20)],
                'd': [random.choice([2, 3, 4]) for _ in range(5)]
            } for _ in range(pop_size)]
            torch.save(pop, init_pop_path)

        print(f"\n=== Calculating {self.dataset_name.upper()} dynamic scaling parameters based on the initial population ===")
        m = [self.get_metrics(a) for a in tqdm(pop, desc="Calibration")]
        hjs, accs = [i[0] for i in m], [i[1] for i in m]

        h_min, h_max = min(hjs), max(hjs)
        a_min, a_max = min(accs) / 100., max(accs) / 100.

        with open(self.parent_history_file, 'a') as f_log:
            for gen in range(generations):
                vault = {}
                for arch in tqdm(pop, desc=f"Gen {gen}"):
                    hj, a5 = self.get_metrics(arch)
                    n_hj = (hj - h_min) / (h_max - h_min + 1e-8)
                    n_a5 = (a5 / 100.0 - a_min) / (a_max - a_min + 1e-8)
                    D = 0.1 * n_hj + n_a5

                    with torch.no_grad():
                        score = predictor.forward(arch, D).item()

                    vault[json.dumps(arch, sort_keys=True)] = {'arch': arch, 'score': score, 'acc': a5}

                top_10 = sorted(vault.values(), key=lambda x: x['score'], reverse=True)[:10]
                print(f"Gen {gen} | Best Acc5e: {top_10[0]['acc']:.2f}%")
                f_log.write(f"Gen {gen} | {top_10[0]['score']:.4f} | {json.dumps(top_10[0]['arch'])}\n")
                f_log.flush()

                pop = [self.mutate(el['arch']) for el in top_10]
                while len(pop) < 10:
                    pop.append(self.mutate(random.choice(top_10)['arch']))

    def mutate(self, arch):
        new_arch = deepcopy(arch)
        orig_str = json.dumps(arch, sort_keys=True)
        for _ in range(100):
            if random.random() < 0.4:
                s_idx = random.randint(0, 4)
                new_arch['d'][s_idx] = random.choice([d for d in self.choices['d'] if d != arch['d'][s_idx]])
            for i in range(20):
                if random.random() < 0.2:
                    new_arch['ks'][i] = random.choice([k for k in self.choices['ks'] if k != arch['ks'][i]])
                if random.random() < 0.2:
                    new_arch['e'][i] = random.choice([e for e in self.choices['e'] if e != arch['e'][i]])
            if json.dumps(new_arch, sort_keys=True) != orig_str: break
        return new_arch


if __name__ == "__main__":
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(42)

    model_dir = '/MobilenetV3/results/predictor/model'
    args.model_path = model_dir
    args.save_path = 'results'
    args.model_name = 'predictor'
    predictor_wrapper = Predictor(args)
    load_model(predictor_wrapper.model, model_dir, load_max_pt='ckpt_max_corr.pt')
    predictor_wrapper.model.cuda().eval()

    nas = UnifiedEvolutionNAS(args.data_name)
    nas.run_search(predictor_wrapper)