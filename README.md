# Particle Swarm Optimization for Multimodal CNN: Early vs. Late Fusion Performance for Action Recognition.

**Extended by** Mihnea Angheluță from the original work by Francisco Erivaldo Fernandes Junior and Gary G. Yen

## Overview

This repository contains an extended implementation of the PSO-CNN algorithm, building upon the work presented in the following paper:

F. E. Fernandes Junior and G. G. Yen, "**Particle swarm optimization of deep neural networks architectures for image classification**," Swarm and Evolutionary Computation, vol. 49, pp. 62–74, Sep. 2019.

```
@article{fernandes_junior_particle_2019,
	title = {Particle swarm optimization of deep neural networks architectures for image classification},
	volume = {49},
	issn = {22106502},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S2210650218309246},
	doi = {10.1016/j.swevo.2019.05.010},
	language = {en},
	urldate = {2019-07-06},
	journal = {Swarm and Evolutionary Computation},
	author = {Fernandes Junior, Francisco Erivaldo and Yen, Gary G.},
	month = sep,
	year = {2019},
	pages = {62--74},
}
```

## Extensions to the Original Work

We have extended the original PSO-CNN implementation with the following features:

1. **Late Fusion Experiments**: We've added a new module (`late_fusion_experiment.py`) that implements late fusion of color and depth streams. This allows for multi-modal fusion where separate CNN architectures are optimized for each modality and then combined.

2. **Early Fusion Support**: Added a .amat conversion file, to convert the preprocessed data into easily digestible data for TensorFlow. The main script now supports early fusion experiments where multi-modal data is processed by a single CNN through the aforementioned preprocessing, essentially implementing the original psoCNN to another dataset.

3. **TensorFlow 2.16.2 Compatibility**: The codebase has been updated to work with TensorFlow 2.16.2 and Keras 3.9.2, incorporating all necessary API changes from the original TensorFlow 1.14 implementation. This was important for Metal Perfomance Shaders (MPS) compatibility on Apple Silicon devices, which were the hardware used for testing.

## Installation

This implementation uses pip for package management instead of the original Anaconda-based setup.

### Prerequisites

- Python 3.11
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/JeffTheNinja57/bachelor_thesis_code.git
   cd [repository-directory]
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   **Note for Apple Silicon (M1+) users**: If you're using an Apple Silicon Mac, uncomment the last two lines in the requirements.txt file to install the Apple-specific TensorFlow packages:
   ```
   # tensorflow-macos~=2.16.2
   # tensorflow-metal~=1.2.0
   ```
3. Dataset is available on https://www.crossvalidate.me/datasets.html

## Usage

### Dataset Preparation

The datasets in .amat format are available upon request.

### Running Early Fusion Experiments

To run an early fusion experiment:

```
# Run the main script
python main.py
```

You can modify the experiment parameters in the `main.py` file, including:
- Dataset selection
- Number of runs and iterations
- Population size
- Batch size and epochs
- Network architecture constraints
- PSO parameters

### Running Late Fusion Experiments

To run a late fusion experiment, which trains separate models for color and depth streams and then combines them:

```
# Run the late fusion experiment script
python late_fusion_experiment.py"
```

You can also modify the `LateFusionPSOCNN` class with your desired parameters in the `late_fusion_experiment.py` file.

## Results Analysis

After running experiments, you can analyze the results using the provided visualization tools:

```
python results_metrics_analysis.py"
```

This will generate a comprehensive report with visualizations of model performance, architecture comparisons, and boxplots of the results.

## Notes

- Unlike the original implementation which only worked with TensorFlow 1.14, this version is compatible with TensorFlow 2.16.2.
- The code has been tested on Apple Silicon (with the appropriate TensorFlow packages).
- You can adjust the hyperparameters in the respective Python files to customize the experiments.

## Acknowledgments

This work builds upon the original PSO-CNN implementation by Fernandes Junior and Yen. We extend our gratitude to the original authors for their foundational work in this area.
