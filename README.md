# ACDP: Architecture-Aware Cross-Dataset Performance Predictor for NAS

![Overview of ACDP:](./pictures/ACDP.png)

## Requirements

```
python == 3.8.20 
pytorch == 2.4.1
torchvision == 0.19.1
scipy == 1.10.1
nasbench == 1.0
nas_201_api == 2.1
```

## Dataset Preparation

This project uses two search spaces: NAS-Bench-201 and MobileNetV3.

**NAS-Bench-201**

Project: https://github.com/D-X-Y/NAS-Bench-201

**MobileNetV3**

Project: https://github.com/mit-han-lab/once-for-all

This project uses six datasets: CIFAR-10, CIFAR-100, MNIST, SVHN, FGVC-Aircraft and Oxford-IIIT Pets.

**CIFAR-10:** https://www.cs.toronto.edu/~kriz/cifar.html

**CIFAR-100:** https://www.cs.toronto.edu/~kriz/cifar.html

**MNIST:** http://yann.lecun.com/exdb/mnist/

**SVHN:** http://ufldl.stanford.edu/housenumbers/

**FGVC-Aircraft:** https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

**Oxford-IIIT Pets:** https://www.robots.ox.ac.uk/~vgg/data/pets/

We provide two ways to access the experimental data:

### 1. Quick Start (Pre-processed Metrics)

All necessary metrics required for our experiments have been pre-extracted and included in the `.data/` directory of each respective search space. You can run the searching and evaluation scripts directly without downloading external datasets.

### 2. Full Training (Raw Datasets)

If you wish to re-evaluate or recalculate the metrics (e.g., hybrid fidelity) from scratch, the raw six datasets are required. Detailed instructions on how to set up the raw datasets and paths can be found in the` ./data/README.md`.

## Experiment

## NAS-Bench-201

The best-trained predictor weights are provided at `./NAS-Bench-201/results/predictor/model/ckpt_max_corr.pt.` You can use them directly for evaluation.

**Train from Scratch**
To retrain the ACDP predictor on the NAS-Bench-201 search space:

```
cd NAS-Bench-201
python main.py --test False
```

**Evaluation & Search**

To evaluate the overall Kendall’s Tau correlation and search for high-performing architectures across different datasets:

```
cd NAS-Bench-201
python main.py --test True --data_name 'cifar10'
```
Note: Options for --data_name include [`cifar100`, `mnist`, `svhn`, `aircraft`, `pets`].

## MobilenetV3

The best-trained predictor weights are provided at `./MobilenetV3/results/predictor/model/ckpt_max_corr.pt.` You can use them directly for evaluation.

**Train from Scratch**
To retrain the ACDP predictor on the MobilenetV3 search space:

```
cd MobilenetV3
python main.py --test False
```

**Evolutionary Architecture Search**
To execute the evolutionary search process across various target datasets:

```
cd MobilenetV3
python ENAS.py --data_name 'cifar10'
```
Note: Options for --data_name include [`cifar100`, `mnist`, `svhn`, `aircraft`, `pets`].

**Full Training & Validation**
To validate the ground-truth performance of the top-1 architectures discovered in each generation:

```
cd MobilenetV3/ENAS
python run.py --data_name 'cifar10'
```
Note: Options for --data_name include [`cifar100`, `mnist`, `svhn`, `aircraft`, `pets`].

**Cross-Dataset Baseline Comparison**
We provide the search trajectories of baseline methods (**MetaD2A** and **DiffusionNAG**) for comparison. To visualize the comparative results:

```
cd MobilenetV3/ENAS
python pic.py --data_name 'cifar10'
```
Note: Options for --data_name include [`cifar100`, `mnist`, `svhn`, `aircraft`, `pets`].
