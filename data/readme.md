## Dataset & Metadata Overview

This repository hosts a comprehensive collection of pre-computed performance metadata, including 5-epoch training accuracies (5acc) and zero-cost proxies (hj), specifically curated for ACDP.

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
Once the raw datasets are in place, you can trigger the hj feature extraction and 5 epoch accuracy computation scripts provided in the `{search_space_root}/data/hybrid-fidelity` directory.
