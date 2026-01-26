import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import glob


from ofa_local.imagenet_classification.networks.mobilenet_v3 import MobileNetV3
from ofa_local.utils import make_divisible
from ofa_local.utils.layers import ConvLayer, MBConvLayer, IdentityLayer, LinearLayer, ResidualBlock

try:
    from scores import get_score_func
except ImportError:
    def get_score_func(score_name):
        if score_name == 'hook_logdet':
            return lambda K: np.linalg.slogdet(K)[1]
        raise NotImplementedError


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_mv3_model(arch_dict, num_classes):
    ks_list, e_list, d_list = arch_dict['ks'], arch_dict['e'], arch_dict['d']
    input_channel, last_channel = 16, 1280
    first_conv = ConvLayer(3, input_channel, kernel_size=3, stride=1, use_bn=True, act_func='relu')

    out_channels = [24, 40, 80, 112, 160, 160, 160, 160]
    ofa_act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
    ofa_se_stages = [False, False, True, False, True, True]
    base_strides = [2, 2, 1, 1, 1]
    stage_strides = [base_strides[i] if i < len(base_strides) else 1 for i in range(len(d_list))]

    blocks, feature_dim, block_idx_in_net = [], input_channel, 0
    for stage_idx in range(len(d_list)):
        depth = d_list[stage_idx]
        out_channel = out_channels[min(stage_idx, len(out_channels) - 1)]
        act_func = ofa_act_stages[min(stage_idx, len(ofa_act_stages) - 1)]
        use_se = ofa_se_stages[min(stage_idx, len(ofa_se_stages) - 1)]
        for block_idx in range(depth):
            ks, e = ks_list[block_idx_in_net], e_list[block_idx_in_net]
            stride = stage_strides[stage_idx] if block_idx == 0 else 1
            mid_channel = make_divisible(feature_dim * e, 8)
            mb_conv = MBConvLayer(feature_dim, out_channel, ks, stride, e, mid_channel, act_func, use_se)
            shortcut = IdentityLayer(out_channel, out_channel) if (stride == 1 and out_channel == feature_dim) else None
            blocks.append(ResidualBlock(mb_conv, shortcut, dropout_rate=0, dropblock=False, block_size=1))
            feature_dim, block_idx_in_net = out_channel, block_idx_in_net + 1

    final_expand_layer = ConvLayer(feature_dim, feature_dim * 6, kernel_size=1, use_bn=True, act_func='h_swish')
    feature_mix_layer = ConvLayer(feature_dim * 6, last_channel, kernel_size=1, bias=False, use_bn=False,act_func='h_swish')
    classifier = LinearLayer(last_channel, num_classes, dropout_rate=0.2)
    return MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)


class MV3ZeroCostFixedBatchProcessor:
    def __init__(self):
        self.data_base_path = '/data'
        self.save_path = 'result'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fixed_batch_size = 128
        self.save_interval = 20

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        db_path = '/MobilenetV3/data/collected_database.pt'
        self.database = torch.load(db_path)
        self.data_len = len(self.database)
        self.img_root_dir = os.path.join(self.data_base_path, 'imagenet32_extracted', 'train')

    def compute_scores(self):
        set_seed(42)
        os.makedirs(self.save_path, exist_ok=True)
        output_full_path = os.path.join(self.save_path, 'imagenet_hj.pt')

        all_task_data = torch.load(output_full_path) if os.path.exists(output_full_path) else {}
        processed_indices = set(all_task_data.keys())

        for index in range(self.data_len):
            if index in processed_indices: continue

            sample = self.database[index]
            raw_classes, arch_dict = sample['class'], sample['net']

            all_task_image_paths = []
            for raw_cls_id in raw_classes:
                cid = int(raw_cls_id.item() if torch.is_tensor(raw_cls_id) else raw_cls_id)
                class_path = os.path.join(self.img_root_dir, str(cid).zfill(3))
                if os.path.isdir(class_path):
                    imgs = glob.glob(os.path.join(class_path, '*.png')) + glob.glob(os.path.join(class_path, '*.jpg'))
                    all_task_image_paths.extend(imgs)

            if len(all_task_image_paths) < self.fixed_batch_size:
                selected_paths = random.choices(all_task_image_paths, k=self.fixed_batch_size)
            else:
                selected_paths = random.sample(all_task_image_paths, self.fixed_batch_size)

            x_batch_list = []
            for p in selected_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    x_batch_list.append(self.transform(img))
                except:
                    continue

            x_batch = torch.stack(x_batch_list).to(self.device)

            net = build_mv3_model(arch_dict, num_classes=len(raw_classes)).to(self.device)
            net.train()
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            hooks = []
            net.K = np.zeros((self.fixed_batch_size, self.fixed_batch_size))

            def counting_forward_hook(module, inp, out):
                nonlocal net
                try:
                    if isinstance(inp, tuple): inp = inp[0]
                    if inp.dim() == 4:
                        inp = torch.mean(inp, dim=[2, 3])

                    inp = inp.view(inp.size(0), -1)
                    x = (inp > 0).float()

                    K = x @ x.t()
                    K2 = (1. - x) @ (1. - x.t())
                    net.K = net.K + (K.cpu().numpy() + K2.cpu().numpy()) / (x.size(1) + 1e-6)
                except:
                    pass

            for name, module in net.named_modules():
                if any(act in str(type(module)) for act in ['ReLU', 'Hswish', 'Hardswish']):
                    hooks.append(module.register_forward_hook(counting_forward_hook))

            try:
                with torch.no_grad():
                    net(x_batch)

                eps = 1e-5
                res = np.linalg.slogdet(net.K + np.eye(self.fixed_batch_size) * eps)
                zero_cost_score = res[1]
                print(f"Task {index + 1} | Batch: {x_batch.shape[0]} | Score: {zero_cost_score:.2f}")
            except Exception as e:
                print(f"Error at Task {index + 1}: {e}")
                zero_cost_score = np.nan
            finally:
                for h in hooks: h.remove()

            all_task_data[index] = {
                'arch_dict': arch_dict,
                'hj': zero_cost_score,
                'top1': sample['top1']
            }

            if (index + 1) % self.save_interval == 0 or (index + 1) == self.data_len:
                torch.save(all_task_data, output_full_path + ".tmp")
                os.replace(output_full_path + ".tmp", output_full_path)

if __name__ == "__main__":
    processor = MV3ZeroCostFixedBatchProcessor()
    processor.compute_scores()