# PSO-CNN Experiment with Comprehensive Metrics

This README provides instructions on how to run the PSO-CNN experiment with comprehensive evaluation metrics and visualization tools.

## Overview

The experiment performs Particle Swarm Optimization (PSO) to search for optimal CNN architectures for action recognition using the iCub dataset. The implementation includes:

1. A dataset loader that correctly handles train/test/val folders
2. Comprehensive evaluation metrics (accuracy, precision, recall, F1-score, etc.)
3. Model efficiency metrics (parameters, size, inference time, FLOPs)
4. Visualization and monitoring tools (TensorBoard, confusion matrices, metrics plots)

## Prerequisites

Before running the experiment, make sure you have:

1. PyTorch installed
2. Required dependencies: numpy, matplotlib, seaborn, scikit-learn, tensorboard, thop
3. The iCub action dataset split into train/test/val folders (use `data_preprocessing/split_dataset.py` if needed)

You can install the required dependencies with:

```bash
pip install torch numpy matplotlib seaborn scikit-learn tensorboard thop
```

## Running the Experiment

To run the experiment with default settings:

```bash
python run_experiment.py
```

### Command-line Arguments

The script supports various command-line arguments to customize the experiment:

#### Dataset Arguments
- `--data_dir`: Directory containing the dataset (default: 'data')
- `--fusion_type`: Fusion type: 'early' or 'late' (default: 'early')

#### Output Arguments
- `--output_dir`: Directory to save results and models (default: 'results')

#### PSO Arguments
- `--swarm_size`: Number of particles in the swarm (default: 10)
- `--max_iter`: Maximum number of PSO iterations (default: 10)
- `--max_layers`: Maximum number of functional layers (default: 5)
- `--gbest_prob`: gBest probability factor (default: 0.7)
- `--max_kernel`: Maximum kernel size (default: 5)
- `--max_maps`: Maximum feature maps per conv layer (default: 64)
- `--max_neurons`: Maximum neurons per FC layer (default: 128)
- `--use_multiprocessing`: Use multiprocessing PSO to evaluate particles in parallel (flag)
- `--num_processes`: Number of processes to use for multiprocessing PSO (default: max(1, cpu_count() - 1))

#### Training Arguments
- `--train_epochs`: Number of epochs for particle evaluation (default: 2)
- `--test_epochs`: Number of epochs for final model training (default: 5)
- `--learning_rate`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 32)
- `--use_bn`: Use batch normalization (flag)
- `--use_dropout`: Use dropout (flag)
- `--dropout_rate`: Dropout rate (default: 0.5)
- `--num_workers`: Number of workers for data loading (default: 4)

#### Device Arguments
- `--device`: Device to use (cpu, cuda, mps). If None, will use the best available.

### Example

To run an experiment with late fusion, a larger swarm, and more training epochs:

```bash
python main.py --fusion_type late --swarm_size 20 --max_iter 15 --test_epochs 10 --use_bn
```

To run an experiment with multiprocessing PSO for faster evaluation:

```bash
python main.py --fusion_type early --use_multiprocessing --num_processes 4
```

## Understanding the Results

The experiment generates several outputs to help you understand and analyze the results:

### 1. Metrics and Results

The experiment calculates and reports the following metrics:

- **Classification Metrics**:
  - Accuracy: Overall accuracy of the model
  - Precision: Precision score (macro-averaged)
  - Recall: Recall score (macro-averaged)
  - F1-score: F1 score (macro-averaged)

- **Model Efficiency Metrics**:
  - Parameters: Number of trainable parameters
  - Model Size: Size of the model in MB
  - Inference Time: Average inference time in milliseconds
  - FLOPs: Floating Point Operations Per Second

All these metrics are saved in a JSON file in the output directory.

### 2. Visualizations

The experiment generates several visualizations:

- **Confusion Matrix**: Shows the model's performance across different classes
- **Metrics History Plots**: Show how metrics evolve during training
- **TensorBoard Logs**: Detailed logs for monitoring training progress

### 3. TensorBoard

To view the TensorBoard logs:

```bash
tensorboard --logdir results/experiment_<fusion_type>_<timestamp>/tensorboard_<fusion_type>_<timestamp>
```

This will start a TensorBoard server that you can access in your web browser at `http://localhost:6006`.

TensorBoard provides interactive visualizations of:
- Training and validation metrics over time
- Confusion matrices
- Model architecture (if available)

### 4. Model Files

The experiment saves the following files:

- **Final Model**: The trained model state dict
- **Configuration**: The experiment configuration in JSON format
- **Results**: Comprehensive evaluation metrics in JSON format

## Interpreting the Metrics

When analyzing the results, consider the following:

1. **Classification Performance**: Higher accuracy, precision, recall, and F1-score indicate better classification performance.

2. **Model Efficiency**: 
   - Lower parameter count and model size indicate a more compact model
   - Lower inference time indicates faster prediction
   - Lower FLOPs indicate more computational efficiency

3. **Trade-offs**: There's often a trade-off between classification performance and model efficiency. The best model depends on your specific requirements.

## Customizing the Experiment

You can customize the experiment by:

1. Modifying the dataset loader in `data_preprocessing/dataset.py`
2. Adding new metrics in `utils/metrics.py`
3. Changing the PSO parameters in `experiment/run.py`
4. Modifying the model architecture search space in the PSO implementation

## Troubleshooting

If you encounter issues:

1. Check that the dataset is correctly structured
2. Ensure all dependencies are installed
3. Try reducing batch size if you encounter memory issues
4. Check the logs for specific error messages
