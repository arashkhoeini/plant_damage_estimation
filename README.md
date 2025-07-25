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

### Option 1: Local Installation

```bash
git clone https://github.com/arashkhoeini/plant_damage_estimation.git
cd plant_damage_estimation
pip install -r requirements.txt
```

### Option 2: Docker Installation (Recommended)

Docker provides a consistent environment with all dependencies pre-installed, including CUDA support for GPU acceleration.

#### Prerequisites
- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)

#### Quick Start with Docker

1. Clone the repository:
```bash
git clone https://github.com/arashkhoeini/plant_damage_estimation.git
cd plant_damage_estimation
```

2. Use the helper script (recommended):
```bash
# Setup directories and check requirements
./docker_helper.sh setup

# Build Docker image
./docker_helper.sh build

# Run training
./docker_helper.sh train

# Run inference (after placing images in inference_input/)
./docker_helper.sh inference

# Start TensorBoard
./docker_helper.sh tensorboard

# Open interactive shell
./docker_helper.sh shell
```

3. Or use Docker Compose directly:
```bash
# Build the Docker image
docker-compose build

# Run training
docker-compose up plant-damage-estimation

# Run inference (create inference_input and inference_output directories first)
mkdir -p inference_input inference_output
docker-compose --profile inference up plant-damage-inference

# Run TensorBoard (optional)
docker-compose --profile tensorboard up tensorboard
```

#### Manual Docker Commands

```bash
# Build the image
docker build -t plant-damage-estimation .

# Run training
docker run --gpus all -v $(pwd)/dataset:/app/dataset -v $(pwd)/saved:/app/saved plant-damage-estimation

# Run inference
docker run --gpus all -v $(pwd)/inference_input:/app/inference_input -v $(pwd)/inference_output:/app/inference_output -v $(pwd)/saved:/app/saved plant-damage-estimation python inference.py -m /app/saved/model.pth -i /app/inference_input -o /app/inference_output
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

## Docker Usage

### Docker Services

The project includes three Docker services:

1. **plant-damage-estimation**: Main training service
2. **plant-damage-inference**: Inference service for running predictions
3. **tensorboard**: TensorBoard visualization service

### Volume Mounts

The Docker setup automatically mounts the following directories:
- `./dataset` → `/app/dataset`: Your dataset directory
- `./saved` → `/app/saved`: Model checkpoints and logs
- `./configs` → `/app/configs`: Configuration files
- `./oracle.pth` → `/app/oracle.pth`: Oracle model file

### GPU Support

The Docker configuration includes NVIDIA GPU support. Make sure you have:
- NVIDIA Docker runtime installed
- Proper GPU drivers

### Environment Variables

Key environment variables in the Docker container:
- `CUDA_VISIBLE_DEVICES=0`: Specify which GPU to use
- `PYTHONPATH=/app`: Python path configuration

### Docker Compose Profiles

Use profiles to run specific services:
```bash
# Training only (default)
docker-compose up

# Inference only
docker-compose --profile inference up plant-damage-inference

# TensorBoard only
docker-compose --profile tensorboard up tensorboard

# Multiple services
docker-compose --profile inference --profile tensorboard up
```

### Customizing Docker Configuration

You can modify the `docker-compose.yml` file to:
- Change GPU device selection
- Modify volume mounts
- Adjust port mappings
- Set different environment variables 

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
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker image configuration
├── docker-compose.yml         # Docker Compose services
├── .dockerignore              # Docker ignore file
└── docker_helper.sh           # Helper script for Docker operations
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
