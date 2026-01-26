import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import copy


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from ofa_local.imagenet_classification.networks.mobilenet_v3 import MobileNetV3
from ofa_local.utils import make_divisible
from ofa_local.utils.layers import set_layer_from_config, MBConvLayer, ConvLayer, IdentityLayer, LinearLayer, ResidualBlock


def build_mv3_model(arch_dict, num_classes):
    ks_list = arch_dict['ks']
    e_list = arch_dict['e']
    d_list = arch_dict['d']

    input_channel = 16
    last_channel = 1280

    first_conv = ConvLayer(
        3, input_channel, kernel_size=3, stride=1, use_bn=True,
        act_func='relu', ops_order='weight_bn_act'
    )

    out_channels = [24, 40, 80, 112, 160, 160, 160, 160]
    ofa_act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
    ofa_se_stages = [False, False, True, False, True, True]

    base_strides = [2, 2, 1, 1, 1]
    stage_strides = [base_strides[i] if i < len(base_strides) else 1 for i in range(len(d_list))]

    blocks = []
    feature_dim = input_channel

    block_idx_in_net = 0

    for stage_idx in range(len(d_list)):
        depth = d_list[stage_idx]
        out_channel = out_channels[min(stage_idx, len(out_channels) - 1)]

        act_func = ofa_act_stages[min(stage_idx, len(ofa_act_stages) - 1)]
        use_se = ofa_se_stages[min(stage_idx, len(ofa_se_stages) - 1)]

        for block_idx in range(depth):
            ks = ks_list[block_idx_in_net]
            e = e_list[block_idx_in_net]

            stride = stage_strides[stage_idx] if block_idx == 0 else 1
            mid_channel = make_divisible(feature_dim * e, 8)

            mb_conv = MBConvLayer(
                feature_dim, out_channel, ks, stride, e, mid_channel, act_func, use_se
            )

            if stride == 1 and out_channel == feature_dim:
                shortcut = IdentityLayer(out_channel, out_channel)
            else:
                shortcut = None

            blocks.append(ResidualBlock(
                mb_conv, shortcut, dropout_rate=0, dropblock=False, block_size=1
            ))

            feature_dim = out_channel
            block_idx_in_net += 1

    final_expand_layer = ConvLayer(
        feature_dim, feature_dim * 6, kernel_size=1, use_bn=True,
        act_func='h_swish', ops_order='weight_bn_act',
    )

    feature_mix_layer = ConvLayer(
        feature_dim * 6, last_channel, kernel_size=1, bias=False,
        use_bn=False, act_func='h_swish',
    )

    classifier = LinearLayer(last_channel, num_classes, dropout_rate=0.2)

    net = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    return net


class MV3FullDatabaseProcessor:
    def __init__(self):
        self.dpath = '/data/'
        self.save_path = 'result'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_epoch = 5
        self.batch_size = 32

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        db_full_path = os.path.join('/MobilenetV3/data', 'collected_database.pt')
        print(f"Loading full database: {db_full_path}")
        self.database = torch.load(db_full_path)

        all_top1s = [item['top1'] for item in self.database]
        all_top1s_tensor = torch.tensor(all_top1s)
        self.global_mean = torch.mean(all_top1s_tensor).item()
        self.global_std = torch.std(all_top1s_tensor).item()

        print(f"Database loading is complete. Total number of samples: {len(self.database)}")

    def run_all(self):
        img_root_dir = os.path.join(self.dpath, 'imagenet32_extracted', 'train')
        os.makedirs(self.save_path, exist_ok=True)
        output_path = os.path.join(self.save_path, 'imagenet_5acc.pt')
        temp_output_path = output_path + ".tmp"

        processed_data = torch.load(output_path) if os.path.exists(output_path) else {}

        for i in range(len(self.database)):
            if i in processed_data: continue
            sample = self.database[i]
            original_top1, raw_classes, arch_dict, flops = sample['top1'], sample['class'], sample['net'], sample['flops']

            print(f'\n[{i + 1}/{len(self.database)}] Process samples (Top1: {original_top1:.4f})...')

            all_file_paths = []
            all_labels = []
            current_local_label = 0

            for raw_cls_id in raw_classes:
                cid = int(raw_cls_id.item() if torch.is_tensor(raw_cls_id) else raw_cls_id)
                class_path = os.path.join(img_root_dir, str(cid).zfill(3))

                if not os.path.exists(class_path): continue

                files = [os.path.join(class_path, f) for f in os.listdir(class_path) if
                         f.lower().endswith(('.png', '.jpg'))]
                if len(files) == 0: continue

                sampled_files = random.sample(files, 10) if len(files) > 10 else files
                all_file_paths.extend(sampled_files)
                all_labels.extend([current_local_label] * len(sampled_files))
                current_local_label += 1

            if not all_file_paths: continue

            class TaskDataset(torch.utils.data.Dataset):
                def __init__(self, paths, labels, transform):
                    self.paths, self.labels, self.transform = paths, labels, transform

                def __len__(self):
                    return len(self.paths)

                def __getitem__(self, idx):
                    try:
                        img = Image.open(self.paths[idx]).convert('RGB')
                        return self.transform(img), self.labels[idx]
                    except:
                        return torch.zeros(3, 32, 32), self.labels[idx]

            train_set = TaskDataset(all_file_paths, all_labels, self.transform)
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
            eval_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False, num_workers=4)


            net = build_mv3_model(arch_dict, num_classes=current_local_label).to(self.device)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

            for epoch in range(self.max_epoch):
                net.train()
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.max_epoch}")
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    pred = net(data)
                    loss = criterion(pred, target)
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                scheduler.step()

            net.eval()
            correct, total, feats = 0, 0, []
            with torch.no_grad():
                for data, target in tqdm(eval_loader, desc="Extracting", leave=False):
                    data, target = data.to(self.device), target.to(self.device)
                    f = net.first_conv(data)
                    for b in net.blocks: f = b(f)
                    f = net.final_expand_layer(f)
                    f = net.global_avg_pool(f)
                    f = net.feature_mix_layer(f)
                    feat = f.view(f.size(0), -1)

                    logits = net.classifier(feat)
                    _, pred = torch.max(logits, 1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                    feats.append(feat.cpu())

            final_accuracy = correct / total
            print(f"Task {i + 1} Final Accuracy: {final_accuracy:.4f}")

            processed_data[i] = {
                'acc': original_top1,
                'class': raw_classes,
                'net': arch_dict,
                'flops': flops,
                'mean': self.global_mean,
                'std': self.global_std,
                'pooled_features': torch.mean(torch.cat(feats, 0), 0, keepdim=True),
                '5_epoch_accuracy': correct / total
            }

            torch.save(processed_data, temp_output_path)
            os.replace(temp_output_path, output_path)


if __name__ == "__main__":
    set_seed(42)
    processor = MV3FullDatabaseProcessor()
    processor.run_all()