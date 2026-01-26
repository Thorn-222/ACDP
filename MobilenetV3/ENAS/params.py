import os
import torch
import torch.nn as nn
from torchprofile import profile_macs

from transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.evaluator import OFAEvaluator


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def count_parameters_direct(arch_config, model_path, n_cls=10, img_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluator = OFAEvaluator(num_gen_arch=None, img_size=32, drop_path=0.2, model_path=model_path)

    evaluator.engine.set_active_subnet(
        ks=arch_config['ks'],
        e=arch_config['e'],
        d=arch_config['d']
    )

    net = evaluator.engine.get_active_subnet()
    if hasattr(net, 'classifier'):
        in_features = net.classifier.linear.in_features
        net.classifier.linear = nn.Linear(in_features, n_cls)

    net = net.to(device)
    net.eval()

    params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6

    inputs = torch.randn(1, 3, img_size, img_size).to(device)
    try:
        flops = profile_macs(net, inputs) / 1e6
    except Exception as e:
        flops = 0.0

    print("-" * 30)
    print(f"Params: {params:.3f} M")
    print(f"FLOPs:  {flops:.2f} M (Input: {img_size}x{img_size})")
    print("-" * 30)

    return params, flops


if __name__ == "__main__":
    OFA_WEIGHTS = '/MobilenetV3/data/ofa_mbv3_d234_e346_k357_w1.0'

    my_arch =   {"ks": [5, 7, 7, 7, 3, 7, 7, 7, 3, 5, 7, 5, 7, 7, 7, 5, 7, 3, 3, 7], "e": [4, 4, 3, 4, 3, 6, 4, 4, 4, 3, 4, 4, 4, 6, 3, 4, 4, 3, 6, 3], "d": [4, 3, 4, 4, 4]}


    count_parameters_direct(my_arch, OFA_WEIGHTS)

