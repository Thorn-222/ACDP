## Dataset & Metadata Overview

This repository hosts a comprehensive collection of pre-computed performance metadata, including 5-epoch training accuracies (**5acc**) and zero-cost proxies (**hj**), specifically curated for ACDP.

All data is serialized and stored within the `.data/` directory of each respective search space. By default, both metrics are provided for immediate use without additional computation; the assets are located as follows:

`{search_space_root}/data/{dataset}_data.pt`

The supported datasets are categorized based on their roles in the ACDP framework:
- **Training Dataset**: `imagenet` (used as the primary domain for ACDP proxy optimization).
- **Unseen Evaluation Datasets**: `cifar10`, `cifar100`, `mnist`, `svhn`, `pets`, and `aircraft` (used to evaluate the generalization capability of ACDP).

## Reproduction & Custom Training
While the repository provides out-of-the-box metadata, we support end-to-end reproduction for users wishing to generate these metrics from scratch.

### 1. Data Prerequisites
To initiate the training or proxy computation pipelines, you must first populate the local environment with the raw datasets. Ensure the data is structured as follows:
```
/data/
├── cifar10/
├── cifar100/
├── MNIST/
├── SVHN/
├── Oxford/
├── FGVC Aircraft/
└── imagenet32_extracted/
```

### 2. Execution Flow
Once the raw datasets are configured, you can compute the hybrid-fidelity features, **5acc** and **hj**, using the scripts provided in the `{search_space_root}/hybrid-fidelity/` directory.

### NAS-Bench-201

We provide separate scripts for the proxy training set (ImageNet) and the six downstream validation datasets.

** 5acc computation**

```
cd Nas-Bench-201/hybrid-fidelity/5acc
# For training set (ImageNet)
python imagenet_5acc.py
# For downstream datasets
python dataset_5acc.py --data_name 'cifar10'
```

** hj computation**

```
cd Nas-Bench-201/hybrid-fidelity/hj
# For training set (ImageNet)
python imagenet_hj.py
# For downstream datasets
python dataset_hj.py --data_name 'cifar10'
```
Note: Options for --data_name include [`cifar100`, `mnist`, `svhn`, `aircraft`, `pets`].

### MobilenetV3

Due to the vast nature of the MobilenetV3 search space, we utilize an evolutionary strategy. For this space, hybrid fidelity is computed on the ImageNet to train ACDP.

** 5acc computation**
```
cd MobilenetV3/hybrid-fidelity
python imagenet_5acc.py
```

** hj computation**
```
cd MobilenetV3/hybrid-fidelity
python imagenet_hj.py
```
