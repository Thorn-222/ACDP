import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import glob
import argparse

DATA_CONFIG = {
    'cifar10': {'classes': 10, 'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2471, 0.2435, 0.2616],'path': 'cifar10'},
    'cifar100': {'classes': 100, 'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761],'path': 'cifar100'},
    'mnist': {'classes': 10, 'mean': [0.1307], 'std': [0.3081], 'path': 'MNIST'},
    'svhn': {'classes': 10, 'mean': [0.4377, 0.4438, 0.4728], 'std': [0.1980, 0.2010, 0.1970], 'path': 'SVHN'},
    'aircraft': {'classes': 100, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],'path': 'FGVC Aircraft'},
    'pets': {'classes': 37, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'path': 'Oxfords'}
}

try:
    from nasbench2 import get_model_from_arch_str
except ImportError:
    print("Warning: Could not import 'get_model_from_arch_str' from 'nasbench2'. "
          "Please ensure 'nasbench2.py' is in your path and contains this function.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class UniversalAccProcessor:
    def __init__(self, ds_name, base_path):
        self.ds_name = ds_name.lower()
        self.cfg = DATA_CONFIG[self.ds_name]
        self.base_path = base_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.max_epoch = 5
        self.save_interval = 30

        arch_path = os.path.join(self.base_path, 'nasbench201_arch.pt')
        self.all_archs = torch.load(arch_path)

        self.out_file = f"result/{self.ds_name}_5acc.pt"
        os.makedirs('result', exist_ok=True)

    def _prepare_dataloader(self):
        print(f"--- Loading data from {self.ds_name.upper()} ---")

        if self.ds_name == 'mnist':
            ts = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(self.cfg['mean'], self.cfg['std'])
            ])
            ds = torchvision.datasets.MNIST(root=self.base_path, train=True, download=True, transform=ts)

        elif self.ds_name == 'svhn':
            ts = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(self.cfg['mean'], self.cfg['std'])
            ])
            ds = torchvision.datasets.SVHN(root=os.path.join(self.base_path, 'svhn'), split='train', download=True,transform=ts)

        else:  # CIFARs, Aircraft, Pets
            ts = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(self.cfg['mean'], self.cfg['std'])
            ])

            img_dir = os.path.join(self.base_path, self.cfg['path'])
            images, labels = [], []

            class_dirs = sorted([d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])
            for idx, c_dir in enumerate(tqdm(class_dirs, desc="Read folder")):
                c_path = os.path.join(img_dir, c_dir)
                files = glob.glob(os.path.join(c_path, '*.*'))
                for f in files:
                    try:
                        img = Image.open(f).convert('RGB')
                        images.append(ts(img))
                        labels.append(idx)
                    except:
                        continue

            ds = TensorDataset(torch.stack(images), torch.tensor(labels))

        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def run(self):
        set_seed(42)
        loader = self._prepare_dataloader()

        all_results = torch.load(self.out_file) if os.path.exists(self.out_file) else {}
        start_idx = 0
        for i, arch in enumerate(self.all_archs):
            if arch in all_results:
                start_idx = i + 1
            else:
                break

        print(f"Starting at index {start_idx}...")

        for index in range(start_idx, len(self.all_archs)):
            arch_str = self.all_archs[index]
            net = get_model_from_arch_str(arch_str, num_classes=self.cfg['classes']).to(self.device)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # train 5 Epoch
            for epoch in range(self.max_epoch):
                net.train()
                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = net(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            net.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = net(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)

            acc = correct / total
            all_results[arch_str] = {'epoch_acc': acc,'arch': arch_str}
            #print(f"[{self.ds_name}] {index + 1}/15625 | Acc: {acc:.4f} | Arch: {arch_str}")

            if (index + 1) % self.save_interval == 0:
                torch.save(all_results, self.out_file + ".tmp")
                os.replace(self.out_file + ".tmp", self.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True, choices=list(DATA_CONFIG.keys()))
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--base_path', type=str, default='/data')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    processor = UniversalAccProcessor(args.dataset, args.base_path)

    processor.run()

