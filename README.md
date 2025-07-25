# Plant Damage Estimation using Deep Learning

A PyTorch implementation for plant damage estimation using deep learning with self-supervised pre-training methods (MoCo and PxCL)

## Overview

This project implements a deep learning approach for estimating plant damage using semantic segmentation. The model supports two self-supervised pre-training methods:
- **MoCo (Momentum Contrast)**: Contrastive learning approach
- **PxCL (Pixel-wise Contrastive Learning)**: Pixel-level contrastive learning

## Features

- Self-supervised pre-training on unlabeled data
- Fine-tuning on labeled plant damage data
- Support for cross-validation evaluation
- Multiple loss functions (CrossEntropy, Lovász, etc.)
- Synchronized batch normalization for multi-GPU training
- Comprehensive evaluation metrics

## Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA (for GPU support)
- See `requirements.txt` for full dependency list

## Installation

```bash
git clone https://github.com/arashkhoeini/plant_damage_estimation.git
cd plant_damage_estimation
pip install -r requirements.txt
```

## Usage

### Configuration

The main configuration file is `configs/configs.yml`. Key parameters include:

- `data_dir`: Path to your dataset
- `num_classes`: Number of segmentation classes (default: 3)
- `pretraining.loss`: Choose `MoCo` or `PxCL` for self-supervised learning
- `finetuning.loss`: Segmentation loss function
- `testing.cross_validation`: Enable k-fold cross-validation

### Training
```bash

python main.py

```

### Inference
```bash
python inference.py -m path/to/model.pth -i input_directory -o output_directory -b 1
```

### Arguments for Inference
- `-m, --model`: Path to trained model (.pth file)
- `-i, --input`: Directory containing input images
- `-o, --output`: Output directory for results
- `-b, --batch`: Batch size for inference

## Oracle
When using `PxCL` pre-training method to use pixel-level constrastive learning, you will need and oracle model to pseudo-label images during pre-training to avoid have false negative pairs. This oracle should be a unet of the same structure, and it could be trained using `MoCo` pre-training followed by supervised finetuning, or just supervised training on the labeled portion of the dataset. For a practical appraoch, first use `MoCo` option to fully pre-train and fune-tune a model, and then use that as your oracle. 

Oracle should be stored as `oracle.pth` in the main directory of the project. [I have prepared a dummy oracle that can be downloaded from here](https://drive.google.com/file/d/1T0nIuFWo79qss2U4JcRu-8ztL_O0HyJI/view?usp=share_link). Note that this oracle is random and it is only useful for test runs. 

## Dataset Structure
There is a dummy dataset available in the `dataset` directory for an easy test run.
The expected dataset structure is:

```
dataset/
├── labeled/
│   ├── fold1/
│   │   ├── fold1_000.png
│   │   ├── fold1_001.png
│   │   ├── ...
│   │   ├── list.txt
│   │   ├── damage_labels/
│   │   │   ├── fold1_000.png
│   │   │   └── ...
│   │   └── leaf_labels/
│   │       ├── fold1_000.png
│   │       └── ...
│   ├── fold2/
│   └── ...
└── unlabeled/
    ├── unlabeled_000.png
    ├── unlabeled_001.png
    ├── ...
    └── list.txt
```

### Data Format
- Images should be in PNG format
- Damage labels: 0 (background), 1 (healthy), 2 (damaged)
- Input images and masks should have the same filename
- Each fold should contain a `list.txt` file with image filenames

## Project Structure

```
plant_damage_estimation_dl/
├── configs/                    # Configuration files
│   ├── configs.yml            # Main configuration
│   └── init_configs.py        # Configuration loader
├── data/                      # Data loading and preprocessing
│   ├── plant_loader.py        # Dataset loader
│   └── prefetcher.py          # Data prefetching utilities
├── models/                    # Model architectures
│   ├── base_model.py          # Base model class
│   ├── resnet.py              # ResNet implementations
│   └── unet.py                # U-Net variants with different backbones
├── utils/                     # Utility functions
│   ├── helpers.py             # General helper functions
│   ├── losses.py              # Loss functions (MoCo, PxCL, etc.)
│   ├── metrics.py             # Evaluation metrics
│   ├── transforms.py          # Data augmentation transforms
│   └── sync_batchnorm/        # Synchronized batch normalization
├── main.py                    # Main training script
├── inference.py               # Inference script
├── trainer.py                 # Training logic
├── oracle.pth                 # pseudo-labeling model, necessary for PxCL
└── requirements.txt           # Python dependencies
```

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@article{khoeini2025plant,
  title={Plant Damage Estimation using Self-Supervised Deep Learning},
  author={Khoeini, Arash},
  journal={To be published},
  year={2025}
}
``` -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Lovász loss implementation adapted from [LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax)
- Synchronized BatchNorm from [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
