import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    from nasbench2 import get_model_from_arch_str
except ImportError:
    print("Warning: Could not import 'get_model_from_arch_str' from 'nasbench2'. "
          "Please ensure 'nasbench2.py' is in your path and contains this function.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed has been set to {seed}")


class ImageFeaturesProcessor:
    def __init__(self):
        self.data_base_path = '/data'
        self.save_path = 'result'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_epoch = 5
        self.batch_size = 128

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.task_lst = torch.load('/Nas-Bench-201/data/meta_train_task_lst.pt')

        data = torch.load('/Nas-Bench-201/data/meta_train_tasks.pt')
        self.acc = data['acc']
        self.task = data['task']
        self.arch = data['arch']

        self.data_len = len(self.acc)

    def pre_train(self):
        set_seed(42)
        img_root_dir = os.path.join(self.data_base_path, 'imagenet32_extracted', 'train')
        if not os.path.exists(img_root_dir):
            raise FileNotFoundError(f"ImageNet-32K data directory not found: {img_root_dir}")

        os.makedirs(self.save_path, exist_ok=True)

        output_feature_file_name = 'imagenet_5acc.pt'
        output_full_path = os.path.join(self.save_path, output_feature_file_name)
        temp_output_full_path = output_full_path + ".tmp"

        all_task_data = {}
        processed_task_indices = set()
        if os.path.exists(output_full_path):
            try:
                all_task_data = torch.load(output_full_path)
                if isinstance(all_task_data, dict):
                    processed_task_indices.update(all_task_data.keys())
                    print(f"Loaded {len(all_task_data)} existing task data from {output_full_path}.")
                else:
                    print(f"Warning: {output_full_path} file contents are not in the correct format and will be started from scratch.")
                    all_task_data = {}
                    processed_task_indices = set()
            except Exception as e:
                print(f"Error loading existing signature file {output_full_path}: {e}.")
                if os.path.exists(temp_output_full_path):
                    try:
                        all_task_data = torch.load(temp_output_full_path)
                        if isinstance(all_task_data, dict):
                            processed_task_indices.update(all_task_data.keys())
                            os.replace(temp_output_full_path, output_full_path)
                            print(f"Data was recovered using a temporary file and renamed {output_full_path}.")
                        else:
                            print(f"Warning: The contents of temporary file {temp_output_full_path} are malformed and will be started from scratch.")
                            all_task_data = {}
                            processed_task_indices = set()
                            if os.path.exists(output_full_path): os.remove(output_full_path)
                            if os.path.exists(temp_output_full_path): os.remove(temp_output_full_path)
                    except Exception as temp_e:
                        all_task_data = {}
                        processed_task_indices = set()
                        if os.path.exists(output_full_path): os.remove(output_full_path)
                        if os.path.exists(temp_output_full_path): os.remove(temp_output_full_path)
                else:
                    print("No temporary files found, will start from scratch.")
                    all_task_data = {}
                    processed_task_indices = set()
                    if os.path.exists(output_full_path): os.remove(output_full_path)

        start_index = 0
        for i in range(self.data_len):
            if i not in processed_task_indices:
                start_index = i
                break
        else:
            print("All tasks have been processed and do not need to be run again.")
            return

        print(f"Processing will start from task index {start_index} ({self.data_len} tasks in total).")

        for index in range(start_index, self.data_len):
            if index in processed_task_indices:
                print(f"Task {index + 1}/{self.data_len} has been processed and skipped.")
                continue

            print(f'Training Task {index + 1}/{self.data_len}:')

            tidx = self.task[index]
            raw_classes = self.task_lst[tidx]

            arch_str = self.arch[index]

            task_images_list = []
            local_labels = []

            for i, raw_cls_idx in enumerate(raw_classes):
                class_folder_name = str(raw_cls_idx).zfill(3)
                class_path = os.path.join(img_root_dir, class_folder_name)

                class CustomImageDataset(torch.utils.data.Dataset):
                    def __init__(self, folder_path, transform=None):
                        self.folder_path = folder_path
                        self.transform = transform
                        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                                            f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                        self.image_files = [f for f in self.image_files if os.path.isfile(f)]
                        self.image_files.sort()

                    def __len__(self):
                        return len(self.image_files)

                    def __getitem__(self, idx):
                        img_path = self.image_files[idx]
                        try:
                            image = Image.open(img_path).convert('RGB')
                            if self.transform:
                                image = self.transform(image)
                            return image
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
                            return torch.zeros(3, 32, 32)

                current_class_dataset = CustomImageDataset(class_path, transform=self.transform)
                class_data_loader = DataLoader(current_class_dataset, batch_size=self.batch_size, shuffle=False,num_workers=4)

                class_images_batch = []
                for images_batch in class_data_loader:
                    valid_images_mask = (images_batch != 0).any(dim=[1, 2, 3]) & (images_batch.shape[1:] == torch.Size([3, 32, 32]))
                    if not valid_images_mask.all():
                        images_batch = images_batch[valid_images_mask]
                        if images_batch.numel() == 0:
                            continue
                    class_images_batch.append(images_batch.cpu())

                if len(class_images_batch) > 0:
                    class_all_images = torch.cat(class_images_batch, dim=0)
                    task_images_list.append(class_all_images)
                    local_labels.extend([i] * class_all_images.shape[0])
                else:
                    print(f"Warning: No valid images loaded for class {raw_cls_idx} in task {index + 1}.")

            if len(raw_classes) != 20:
                print(f"Warning: The number of classes for task {index+1} should be 20. Found {len(raw_classes)}. Continuing anyway.")

            if len(task_images_list) == 0:
                print(f"Skipping task {index + 1} due to no images loaded from any class.")
                continue

            x_data = torch.cat(task_images_list, dim=0)
            y = torch.tensor(local_labels, dtype=torch.long)

            task_data_loader_nas = DataLoader(TensorDataset(x_data, y), batch_size=self.batch_size, shuffle=True)
            eval_data_loader_nas = DataLoader(TensorDataset(x_data, y), batch_size=self.batch_size, shuffle=True)

            net = get_model_from_arch_str(arch_str, num_classes=len(raw_classes)).to(self.device)

            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(self.max_epoch):
                net.train()
                with tqdm(total=len(task_data_loader_nas),
                          desc=f"Epoch {epoch + 1}/{self.max_epoch} (Task {index + 1})", unit="batch") as pbar:
                    for batch_idx, (data, target) in enumerate(task_data_loader_nas):
                        if data.shape[0] == 1:
                            data = torch.cat([data, data], dim=0)
                            target = torch.cat([target, target], dim=0)

                        data = data.to(self.device)
                        target = target.to(self.device)

                        optimizer.zero_grad()

                        output, features = net(data)

                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                        pbar.set_postfix(loss=loss.item())
                        pbar.update(1)

            net.eval()
            with torch.no_grad():
                correct_predictions = 0
                total_predictions = 0

                for batch_data, batch_target in eval_data_loader_nas:
                    batch_data = batch_data.to(self.device)
                    batch_target = batch_target.to(self.device)

                    batch_output  = net(batch_data)

                    _, predicted = torch.max(batch_output, 1)
                    correct_predictions += (predicted == batch_target).sum().item()
                    total_predictions += batch_target.size(0)


                final_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
                print(f"Task {index + 1} (architecture: {arch_str}) Final accuracy (5 epoch): {final_accuracy:.4f}")


                all_task_data[index] = {
                    'arch': arch_str,
                    'tidx': tidx,
                    'epoch_acc': final_accuracy
                }

                processed_task_indices.add(index)

                try:
                    torch.save(all_task_data, temp_output_full_path)
                    os.replace(temp_output_full_path, output_full_path)
                    print(f"Task {index + 1} data (features and accuracy) has been safely saved to {output_full_path}.")
                except Exception as save_e:
                    print(f"An error occurred while saving data for task {index + 1}: {save_e}. The data may not have been completely saved.")
                    if os.path.exists(temp_output_full_path):
                        os.remove(temp_output_full_path)


if __name__ == "__main__":
    if not os.path.exists('result'):
        os.makedirs('result')

    features_processor = ImageFeaturesProcessor()
    features_processor.pre_train()