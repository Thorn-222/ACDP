import os
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import glob
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    from nasbench2 import get_model_from_arch_str
except ImportError:
    print("Warning: Could not import 'get_model_from_arch_str' from 'nasbench2'. "
          "Please ensure 'nasbench2.py' is in your path and contains this function.")

try:
    from scores import get_score_func
except ImportError:
    print("Error: Could not import 'get_score_func' from 'scores.py'. Please ensure 'scores.py' is in your path.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"The random seed has been set to {seed}")


class ZeroCostScoreProcessor:
    def __init__(self):
        self.data_base_path = '/data'
        self.save_path = 'result'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.save_interval = 30

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.task_lst = torch.load('/Nas-Bench-201/data/meta_train_task_lst.pt')
        data = torch.load('/Nas-Bench-201/data/meta_train_tasks.pt')
        self.task = data['task']
        self.arch = data['arch']
        self.data_len = len(self.task)

        self.img_root_dir = os.path.join(self.data_base_path, 'imagenet32_extracted', 'train')
        if not os.path.exists(self.img_root_dir):
            raise FileNotFoundError(f"ImageNet-32K data catalog not found: {self.img_root_dir}")

        print(f"Successfully loaded {self.data_len} meta-training tasks.")

    def compute_zero_cost_scores(self):
        set_seed(42)

        os.makedirs(self.save_path, exist_ok=True)

        output_feature_file_name = 'imagenet_hj.pt'
        output_full_path = os.path.join(self.save_path, output_feature_file_name)
        temp_output_full_path = output_full_path + ".tmp"

        all_task_data = {}
        processed_task_indices = set()

        if os.path.exists(output_full_path):
            try:
                all_task_data = torch.load(output_full_path)
                if isinstance(all_task_data, dict):
                    processed_task_indices.update(all_task_data.keys())
                    print(f"The zero-cost scores of {len(all_task_data)} existing tasks have been loaded from {output_full_path}.")
                else:
                    print(f"Warning: The file {output_full_path} is not formatted correctly. It will start from the beginning.")
                    all_task_data = {}
                    processed_task_indices = set()
            except Exception as e:
                print(f"Error loading existing score file {output_full_path}: {e}. Starting from scratch.")
                all_task_data = {}
                processed_task_indices = set()

        start_index = 0
        for i in range(self.data_len):
            if i not in processed_task_indices:
                start_index = i
                break
        else:
            print("All tasks have been completed and do not need to be run again.")
            return

        print(f"Processing will begin from task index {start_index} (a total of {self.data_len} tasks).")

        for index in range(start_index, self.data_len):
            if index in processed_task_indices:
                print(f"The task {index + 1}/{self.data_len} has been processed and will be skipped.")
                continue

            print(f'Calculating Score for Task {index + 1}/{self.data_len}:')

            tidx = self.task[index]
            raw_classes = self.task_lst[tidx]  # 20 classes for the current task
            arch_str = self.arch[index]
            num_classes_task = len(raw_classes)

            all_task_image_paths = []

            for raw_cls_idx in raw_classes:
                class_folder_name = str(raw_cls_idx).zfill(3)
                class_path = os.path.join(self.img_root_dir, class_folder_name)

                if os.path.isdir(class_path):
                    image_files = glob.glob(os.path.join(class_path, '*.png')) + \
                                  glob.glob(os.path.join(class_path, '*.jpg')) + \
                                  glob.glob(os.path.join(class_path, '*.jpeg')) + \
                                  glob.glob(os.path.join(class_path, '*.gif')) + \
                                  glob.glob(os.path.join(class_path, '*.bmp'))
                    all_task_image_paths.extend(image_files)

            if len(all_task_image_paths) < self.batch_size:
                print(f"Error: Not enough images ({len(all_task_image_paths)}) found for Task {index + 1}. Skipping.")
                all_task_data[index] = {'arch_str': arch_str, 'tidx': tidx, 'zero_cost_score': np.nan}
                processed_task_indices.add(index)
                continue

            selected_paths = random.sample(all_task_image_paths, self.batch_size)
            x_data_list = []

            for img_path in tqdm(selected_paths, desc=f"Loading Batch for Task {index + 1}", unit="image"):
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.transform(image)
                    x_data_list.append(image_tensor)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}. Skipping.")

            if len(x_data_list) != self.batch_size:
                print(f"Warning: Loaded less than {self.batch_size} images due to errors. Skipping Task {index + 1}.")
                all_task_data[index] = {'arch_str': arch_str, 'tidx': tidx, 'zero_cost_score': np.nan}
                processed_task_indices.add(index)
                continue

            x_zero_cost = torch.stack(x_data_list, dim=0).to(self.device)

            net = get_model_from_arch_str(arch_str, num_classes=num_classes_task).to(self.device)
            net.train()
            zero_cost_score = np.nan
            hooks = []

            try:
                net.K = np.zeros((self.batch_size, self.batch_size))

                def counting_forward_hook(module, inp, out):
                    nonlocal net
                    try:
                        if isinstance(inp, tuple):
                            inp = inp[0]
                        inp = inp.view(inp.size(0), -1)

                        x = (inp > 0).float()

                        K = x @ x.t()
                        K2 = (1. - x) @ (1. - x.t())

                        net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()
                    except:
                        pass

                for name, module in net.named_modules():
                    if 'ReLU' in str(type(module)):
                        hooks.append(module.register_forward_hook(counting_forward_hook))

                with torch.no_grad():
                    net(x_zero_cost)

                if hasattr(net, 'K') and net.K.shape[0] == self.batch_size:
                    hooklogdet_func = get_score_func('hook_logdet')
                    zero_cost_score = hooklogdet_func(net.K)

                print(f"Zero-Cost hook_logdet Score: {zero_cost_score:.4f}")

            except Exception as e:
                print(f"Error in Zero-Cost Score calculation for Task {index + 1} ({arch_str}): {e}")
                zero_cost_score = np.nan

            finally:
                for h in hooks:
                    h.remove()

            all_task_data[index] = {
                'arch_str': arch_str,
                'tidx': tidx,
                'hj': zero_cost_score,
            }

            processed_task_indices.add(index)

            if (index + 1) % self.save_interval == 0 or (index + 1) == self.data_len:
                print(f"--- Batch save: Saving zero-cost scores for {len(all_task_data)} tasks ---")
                try:
                    torch.save(all_task_data, temp_output_full_path)
                    os.replace(temp_output_full_path, output_full_path)
                    print(f"The data has been safely saved to {output_full_path}.")
                except Exception as save_e:
                    print(f"An error occurred while saving data in batches: {save_e}. The data may not have been completely saved.")
                    if os.path.exists(temp_output_full_path):
                        os.remove(temp_output_full_path)


if __name__ == "__main__":
    if not os.path.exists('result'):
        os.makedirs('result')

    processor = ZeroCostScoreProcessor()
    processor.compute_zero_cost_scores()