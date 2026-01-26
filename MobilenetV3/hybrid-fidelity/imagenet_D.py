import os
import torch
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt


def normalize_and_combine_and_save(hj_path, acc5e_path, save_root='/MobilenetV3/data/'):
    data_hj = torch.load(hj_path) if os.path.exists(hj_path) else {}
    data_5e = torch.load(acc5e_path) if os.path.exists(acc5e_path) else {}

    common_indices = set(data_hj.keys()) & set(data_5e.keys())
    if not common_indices:
        print("Common index not found, please check file contents.")
        return

    print(f"Total number of samples processed: {len(common_indices)}")

    raw_hj = []
    raw_5e = []
    raw_gt = []
    arch_list = []

    for idx in sorted(list(common_indices)):
        if 'hj' not in data_hj[idx] or '5_epoch_accuracy' not in data_5e[idx]:
            continue

        h = data_hj[idx]['hj']
        a = data_5e[idx]['5_epoch_accuracy']
        g = data_5e[idx].get('acc', data_5e[idx].get('top1'))
        arch = data_5e[idx]['net']

        if np.isnan(h) or np.isnan(a): continue

        raw_hj.append(h)
        raw_5e.append(a)
        raw_gt.append(g)
        arch_list.append(arch)

    raw_hj = np.array(raw_hj)
    raw_5e = np.array(raw_5e)
    raw_gt = np.array(raw_gt)

    norm_hj = (raw_hj - raw_hj.min()) / (raw_hj.max() - raw_hj.min() + 1e-8)
    norm_5e = (raw_5e - raw_5e.min()) / (raw_5e.max() - raw_5e.min() + 1e-8)

    D = norm_hj + norm_5e

    tau_D, _ = kendalltau(D, raw_gt)
    print(f"\nKendall Tau fusing indicator D with Ground Truth: {tau_D:.4f}")

    combined_data = {
        'acc': raw_gt.tolist(),
        'arch': arch_list,
        'D': D.tolist()
    }

    mv3_data_path = os.path.join(save_root, 'imagenet_data.pt')
    torch.save(combined_data, mv3_data_path)

    plt.scatter(D, raw_gt, alpha=0.3)
    plt.xlabel("Combined Metric D")
    plt.ylabel("Ground Truth Top1")
    plt.savefig("check_d_distribution.png")
    plt.close()


if __name__ == "__main__":
    hj_file = '/MobilenetV3/hybrid-fidelity/result/imagenet_hj.pt'
    acc5e_file = '/MobilenetV3/hybrid-fidelity/result/imagenet_5acc.pt'

    target_dir = '/MobilenetV3/data/'
    normalize_and_combine_and_save(hj_file, acc5e_file, save_root=target_dir)