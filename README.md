# Multimodal PSO-CNN for iCub Action Recognition

This repository contains the code for a multimodal Particle Swarm Optimization (PSO) based Convolutional Neural Network (CNN) architecture search for action recognition on the iCub dataset.

## Project Overview

This project implements a PSO-based approach to automatically search for optimal CNN architectures for action recognition tasks. The key features include:

- **Multimodal Fusion**: Support for both early and late fusion strategies to combine color and depth information
- **PSO-based Architecture Search**: Automatic search for optimal CNN architectures using Particle Swarm Optimization
- **iCub Action Dataset**: Designed to work with the iCub humanoid robot action recognition dataset
- **Flexible CNN Building**: Dynamic construction of CNN architectures based on PSO-generated encodings

## Repository Structure

- **data_preprocessing/**: Scripts for data preparation and dataset implementation
  - `dataset.py`: Implementation of the ActionDataset class for data loading
  - `concat_images.py`: Tools for concatenating images for early fusion
  - `split_dataset.py`: Utilities for splitting datasets into train/val/test
  - `preprocessing.py`: General preprocessing functions
  - `check_dimensions.py`: Tools for checking image dimensions

- **models/**: Neural network model implementations
  - `cnn_architecture.py`: Dynamic CNN architecture builder
  - `fusion_models.py`: Early and late fusion model implementations

- **pso/**: Particle Swarm Optimization implementation
  - `pso.py`: Main PSO algorithm implementation
  - `pso_helpers.py`: Helper functions for PSO
  - `particle.py`: Particle class definition
  - `initialize_swarm.py`: Swarm initialization functions

- **experiment/**: Experiment running code
  - `run.py`: Main experiment runner

- **utils/**: Utility functions
  - `train.py`: Training and evaluation functions
  - `helpers.py`: General helper functions

- **docs/**: Documentation
  - `tasks.md`: Current task list
  - `updated_tasks.md`: Comprehensive improvement checklist

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/thesis_code.git
   cd thesis_code
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running Experiments

To run a PSO-CNN experiment:

```bash
python main.py --data_dir /path/to/dataset --fusion_type early --output_dir results
```

Key parameters:
- `--data_dir`: Path to the dataset directory
- `--fusion_type`: Fusion strategy to use ('early' or 'late')
- `--output_dir`: Directory to save results
- `--swarm_size`: Number of particles in the swarm
- `--max_iter`: Maximum number of PSO iterations
- `--e_train`: Epochs for particle evaluation during PSO
- `--e_test`: Epochs for final training of the best model

For a complete list of parameters, run:
```bash
python main.py --help
```

## Dataset

The code is designed to work with the iCub action recognition dataset, which contains color and depth image pairs of various actions performed by the iCub humanoid robot. The dataset should be organized as follows:

- For early fusion:
  ```
  data/
  └── early/
      ├── train/
      ├── val/
      └── test/
  ```

- For late fusion:
  ```
  data/
  ├── late_color/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── late_depth/
      ├── train/
      ├── val/
      └── test/
  ```

## Future Improvements

See the [updated_tasks.md](docs/updated_tasks.md) file for a comprehensive list of planned improvements and features.

## License

[Specify your license here]

## Acknowledgments

- [Acknowledge any resources, papers, or libraries that were particularly helpful]