import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import glob
import argparse


try:
    from nasbench2 import get_model_from_arch_str
except ImportError as e:
    print(f"Error: Missing necessary local file nasbench2.py: {e}")

try:
    from scores import get_score_func
except ImportError:
    print("Error: Could not import 'get_score_func' from 'scores.py'. Please ensure 'scores.py' is in your path.")

    def get_score_func(score_name):
        raise NotImplementedError("get_score_func is not available.")


DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2471, 0.2435, 0.2616],
        'relative_path': 'cifar10',
        'save_name': 'cifar10_hj.pt'
    },
    'cifar100': {
        'num_classes': 100,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
        'relative_path': 'cifar100',
        'save_name': 'cifar100_hj.pt'
    },
    'mnist': {
        'num_classes': 10,
        'mean': [0.1307],
        'std': [0.3081],
        'relative_path': 'MNIST',
        'save_name': 'mnist_hj.pt'
    },
    'svhn': {
        'num_classes': 10,
        'mean': [0.4377, 0.4438, 0.4728],
        'std': [0.1980, 0.2010, 0.1970],
        'relative_path': 'SVHN',
        'save_name': 'svhn_hj.pt'
    },
    'aircraft': {
        'num_classes': 100,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'relative_path': 'FGVC Aircraft',
        'save_name': 'aircraft_hj.pt'
    },
    'pets': {
        'num_classes': 37,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'relative_path': 'Oxford',
        'save_name': 'pets_hj.pt'
    }
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"The random seed has been set to {seed}")


class UniversalZeroCostProcessor:
    def __init__(self, dataset_name, dataset_path):
        self.ds_name = dataset_name.lower()
        if self.ds_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {self.ds_name}")

        self.cfg = DATASET_CONFIGS[self.ds_name]
        self.data_base_path = dataset_path
        self.save_path = 'result'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.save_interval = 30

        self.arch_pt_path = os.path.join('Nas-Bench-201/data', 'nasbench201_arch.pt')
        self.all_arch_strings = torch.load(self.arch_pt_path)

        self.zero_cost_batch = self._prepare_data_batch().to(self.device)

    def _prepare_data_batch(self):
        print(f"--- Preparing a batch of data for {self.ds_name} ---")

        # MNIST
        if self.ds_name == 'mnist':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(self.cfg['mean'], self.cfg['std'])
            ])
            ds = torchvision.datasets.MNIST(root=self.data_base_path, train=True, download=True, transform=transform)
            indices = random.sample(range(len(ds)), self.batch_size)
            return torch.stack([ds[i][0] for i in indices])

        # SVHN
        elif self.ds_name == 'svhn':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(self.cfg['mean'], self.cfg['std'])
            ])
            ds = torchvision.datasets.SVHN(root=self.data_base_path, split='train', download=True, transform=transform)
            indices = random.sample(range(len(ds)), self.batch_size)
            return torch.stack([ds[i][0] for i in indices])

        # CIFAR, Aircraft, Pets
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(self.cfg['mean'], self.cfg['std'])
            ])

            img_dir = os.path.join(self.data_base_path, self.cfg['relative_path'])
            all_files = []
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                all_files.extend(glob.glob(os.path.join(img_dir, '**', ext), recursive=True))

            if not all_files:
                raise FileNotFoundError(f"No image files found under {img_dir}")

            selected = random.sample(all_files, min(len(all_files), self.batch_size))
            x_list = [transform(Image.open(f).convert('RGB')) for f in selected]

            while len(x_list) < self.batch_size:
                x_list.append(torch.zeros(3, 32, 32))

            return torch.stack(x_list)

    def compute(self):
        set_seed(42)
        os.makedirs(self.save_path, exist_ok=True)
        output_path = os.path.join(self.save_path, self.cfg['save_name'])

        all_arch_scores = torch.load(output_path) if os.path.exists(output_path) else {}

        start_idx = 0
        processed_set = set(all_arch_scores.keys())
        for i, arch in enumerate(self.all_arch_strings):
            if arch in processed_set:
                start_idx = i + 1
            else:
                break

        print(f"Starting from index {start_idx}, calculate {self.ds_name}...")

        for index in range(start_idx, len(self.all_arch_strings)):
            arch_str = self.all_arch_strings[index]

            net = get_model_from_arch_str(arch_str, num_classes=self.cfg['num_classes']).to(self.device)
            net.train()

            #  Zero-Cost (hook_logdet)
            net.K = np.zeros((self.batch_size, self.batch_size))
            hooks = []

            def hook_fn(module, inp, out):
                nonlocal net
                try:
                    x = inp[0].view(inp[0].size(0), -1)
                    mask = (x > 0).float()
                    K = mask @ mask.t() + (1. - mask) @ (1. - mask.t())
                    net.K += K.cpu().numpy()
                except:
                    pass

            for module in net.modules():
                if 'ReLU' in str(type(module)):
                    hooks.append(module.register_forward_hook(hook_fn))

            with torch.no_grad():
                net(self.zero_cost_batch)

            for h in hooks: h.remove()

            try:
                score_func = get_score_func('hook_logdet')
                final_score = score_func(net.K)
            except Exception as e:
                print(f"Computational architecture {index} score failed: {e}")
                final_score = np.nan

            all_arch_scores[arch_str] = {'hj': final_score,'arch':arch_str}

            #print(f"[{self.ds_name}] {index + 1}/15625 | Score: {final_score:.4f} | Arch: {arch_str}")

            if (index + 1) % self.save_interval == 0 or (index + 1) == len(self.all_arch_strings):
                torch.save(all_arch_scores, output_path + ".tmp")
                os.replace(output_path + ".tmp", output_path)
                print(f"--- Progress saved to {output_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NAS-Bench-201 hook_logdet metric calculation script')
    parser.add_argument('--data_name', type=str, required=True,help='cifar10, cifar100, mnist, svhn, aircraft, pets')
    parser.add_argument('--base_path', type=str, default='data', help='Dataset root directory')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    processor = UniversalZeroCostProcessor(args.data_name, args.base_path)

    processor.compute()

