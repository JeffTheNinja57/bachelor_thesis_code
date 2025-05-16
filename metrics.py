import datetime
import logging
import os
import time
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class ResultsManager:
    """
    Handles experiment result storage, metric calculations, and visualization.

    This class provides a comprehensive set of utilities for managing machine learning experiment results.
    It handles directory creation, saving models, metrics, plots, and generating detailed reports.
    The class is designed to organize results in a structured way and provide consistent interfaces
    for result analysis and visualization.
    """

    def __init__(self, dataset_name: str, experiment_prefix: Optional[str] = None, include_timestamp: bool = True):
        """
        Initialize the results manager with experiment details.

        :param dataset_name: Name of the dataset used for the experiment
        :type dataset_name: str
        :param experiment_prefix: Optional prefix for the experiment (e.g., 'early', 'late')
        :type experiment_prefix: Optional[str]
        :param include_timestamp: Whether to include a timestamp in the results directory
        :type include_timestamp: bool
        """
        self.dataset_name = dataset_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create experiment directory name based on settings
        if experiment_prefix:
            self.experiment_name = f"{experiment_prefix}_{dataset_name}"
        else:
            self.experiment_name = dataset_name

        if include_timestamp:
            self.experiment_dir = f"./results/experiment_{self.experiment_name}_{self.timestamp}"
        else:
            self.experiment_dir = f"./results/{self.experiment_name}"

        # Create the results directory and subdirectories
        self._create_directories()

        # Initialize metrics storage
        self.metrics_history = {}

        log.info(f"Results will be saved to: {self.experiment_dir}")

    def _create_directories(self):
        """
        Create necessary directories for storing results.

        This method creates the main experiment directory and subdirectories for plots,
        models, and metrics. It's called during initialization of the ResultsManager.

        :return: None
        :rtype: None
        """
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/models", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/metrics", exist_ok=True)

    def get_results_path(self, subdir: Optional[str] = None) -> str:
        """
        Get the full path to the results directory or subdirectory.

        This method constructs and returns the full path to either the main experiment
        directory or a specified subdirectory within it.

        :param subdir: Optional subdirectory name
        :type subdir: Optional[str]
        :return: Full path to the directory
        :rtype: str
        """
        if subdir:
            return os.path.join(self.experiment_dir, subdir)
        return self.experiment_dir

    def save_plot(self, data: np.ndarray, filename: str, xlabel: str = 'Iteration', ylabel: str = 'Accuracy',
                  title: str = None, run_idx: Optional[int] = None):
        """
        Save a plot of data to the plots directory.

        This method creates a line plot of the provided data and saves it as a PNG file
        in the plots directory. It automatically handles file naming based on the provided
        filename and optional run index.

        :param data: Array of data to plot
        :type data: np.ndarray
        :param filename: Base filename without extension
        :type filename: str
        :param xlabel: Label for x-axis
        :type xlabel: str
        :param ylabel: Label for y-axis
        :type ylabel: str
        :param title: Plot title (if None, a default title is generated)
        :type title: str or None
        :param run_idx: Optional run index to append to filename
        :type run_idx: Optional[int]
        :return: Path to the saved plot file
        :rtype: str
        """
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if title:
            plt.title(title)
        else:
            plt.title(f"{self.experiment_name} - {ylabel} vs {xlabel}")

        plt.grid(True)

        # Create filename with optional run index
        if run_idx is not None:
            full_filename = f"{filename}_run_{run_idx}.png"
        else:
            full_filename = f"{filename}.png"

        # Save to plots directory
        filepath = os.path.join(self.get_results_path("plots"), full_filename)
        plt.savefig(filepath)
        plt.close()
        log.info(f"Plot saved to {filepath}")

        return filepath

    def save_model(self, model, name: str, run_idx: Optional[int] = None) -> str:
        """
        Save a model to the models directory.

        This method saves a Keras model to the models directory with the specified name.
        It automatically handles file naming based on the provided name and optional run index.
        The model is saved in the Keras format (.keras).

        :param model: Keras model to save
        :type model: tf.keras.Model
        :param name: Base name for the saved model
        :type name: str
        :param run_idx: Optional run index to append to filename
        :type run_idx: Optional[int]
        :return: Path to the saved model
        :rtype: str
        """
        if run_idx is not None:
            full_name = f"{name}_run_{run_idx}.keras"
        else:
            full_name = f"{name}.keras"

        filepath = os.path.join(self.get_results_path("models"), full_name)
        model.save(filepath)
        log.info(f"Model saved to {filepath}")

        return filepath

    def save_metrics(self, metrics: Dict[str, Any], filename: str, run_idx: Optional[int] = None) -> str:
        """
        Save metrics to a numpy file.

        This method saves a dictionary of metrics to a numpy file in the metrics directory.
        It automatically handles file naming based on the provided filename and optional run index.
        The metrics are saved in the NumPy format (.npy) for easy loading in future analysis.

        :param metrics: Dictionary of metric values
        :type metrics: Dict[str, Any]
        :param filename: Base filename without extension
        :type filename: str
        :param run_idx: Optional run index to append to filename
        :type run_idx: Optional[int]
        :return: Path to the saved metrics file
        :rtype: str
        """
        if run_idx is not None:
            full_filename = f"{filename}_run_{run_idx}.npy"
        else:
            full_filename = f"{filename}.npy"

        filepath = os.path.join(self.get_results_path("metrics"), full_filename)
        np.save(filepath, metrics)
        log.info(f"Metrics saved to {filepath}")

        return filepath

    def save_text_summary(self, text: str, filename: str, run_idx: Optional[int] = None) -> str:
        """
        Save a text summary to the metrics directory.

        This method saves a text summary to a file in the metrics directory.
        It automatically handles file naming based on the provided filename and optional run index.
        The summary is saved as a plain text file (.txt) for easy reading and sharing.

        :param text: Text content to save
        :type text: str
        :param filename: Base filename without extension
        :type filename: str
        :param run_idx: Optional run index to append to filename
        :type run_idx: Optional[int]
        :return: Path to the saved text file
        :rtype: str
        """
        if run_idx is not None:
            full_filename = f"{filename}_run_{run_idx}.txt"
        else:
            full_filename = f"{filename}.txt"

        filepath = os.path.join(self.get_results_path("metrics"), full_filename)
        with open(filepath, "w") as f:
            f.write(text)
        log.info(f"Text summary saved to {filepath}")

        return filepath

    def evaluate_model(self, model, x_test, y_test, batch_size: int = 32, training_time: float = None) -> Tuple[
        Dict[str, Any], str]:
        """
        Evaluate a model on multiple metrics for comprehensive analysis.

        This method performs a thorough evaluation of a Keras model, calculating various
        performance metrics including accuracy, precision, recall, F1 score, and confusion matrix.
        It also measures inference time and collects information about model parameters.
        The results are returned both as a structured dictionary and as a formatted text summary.

        :param model: Keras model to evaluate
        :type model: tf.keras.Model
        :param x_test: Test input data
        :type x_test: np.ndarray
        :param y_test: Test target data (one-hot encoded)
        :type y_test: np.ndarray
        :param batch_size: Batch size for evaluation
        :type batch_size: int
        :param training_time: Optional training time to include in metrics
        :type training_time: float or None
        :return: Tuple containing (metrics_dict, metrics_summary_text)
        :rtype: Tuple[Dict[str, Any], str]
        """
        metrics = {}

        # Record start time for inference timing
        start_time = time.time()
        y_pred = model.predict(x_test, batch_size=batch_size)
        inference_time = time.time() - start_time

        # Convert predictions and labels to class indices
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Calculate accuracy
        accuracy = np.mean(y_pred_classes == y_true_classes)

        # Calculate precision, recall, and F1 score
        precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

        # Generate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        # Calculate model parameter counts
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_count + non_trainable_count

        # Save all metrics
        metrics['accuracy'] = accuracy
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        metrics['confusion_matrix'] = cm
        metrics['trainable_params'] = trainable_count
        metrics['non_trainable_params'] = non_trainable_count
        metrics['total_params'] = total_params
        metrics['inference_time'] = inference_time
        metrics['inference_time_per_sample'] = inference_time / len(x_test)

        if training_time is not None:
            metrics['training_time'] = training_time

        # Generate text summary of metrics
        metrics_summary = "\nMODEL EVALUATION METRICS\n" + "=" * 50 + "\n"
        metrics_summary += f"\nQUANTITATIVE METRICS:\n"
        metrics_summary += f"Accuracy: {metrics['accuracy']:.4f}\n"
        metrics_summary += f"Precision: {metrics['precision']:.4f}\n"
        metrics_summary += f"Recall: {metrics['recall']:.4f}\n"
        metrics_summary += f"F1 Score: {metrics['f1_score']:.4f}\n"

        metrics_summary += f"\nQUALITATIVE METRICS:\n"
        metrics_summary += f"Total Parameters: {metrics['total_params']:,}\n"
        metrics_summary += f"Trainable Parameters: {metrics['trainable_params']:,}\n"
        metrics_summary += f"Non-Trainable Parameters: {metrics['non_trainable_params']:,}\n"
        metrics_summary += f"Inference Time (total): {metrics['inference_time']:.4f} seconds\n"
        metrics_summary += f"Inference Time (per sample): {metrics['inference_time_per_sample'] * 1000:.4f} ms\n"

        if 'training_time' in metrics:
            metrics_summary += f"Training Time: {metrics['training_time']:.2f} seconds\n"

        return metrics, metrics_summary

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str] = None,
                              run_idx: Optional[int] = None) -> str:
        """
        Plot and save a confusion matrix.

        This method creates a heatmap visualization of a confusion matrix and saves it
        as a PNG file in the plots directory. If class names are not provided, it uses
        numbered classes. The plot uses a blue color scheme with annotations showing
        the count in each cell.

        :param cm: Confusion matrix as numpy array
        :type cm: np.ndarray
        :param class_names: List of class names (optional)
        :type class_names: List[str] or None
        :param run_idx: Optional run index to append to filename
        :type run_idx: Optional[int]
        :return: Path to the saved plot
        :rtype: str
        """
        plt.figure(figsize=(10, 8))

        if class_names is None:
            # Use numbered classes if no names provided
            class_names = [str(i) for i in range(cm.shape[0])]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f"{self.experiment_name} - Confusion Matrix")

        if run_idx is not None:
            filename = f"confusion_matrix_run_{run_idx}.png"
        else:
            filename = "confusion_matrix.png"

        filepath = os.path.join(self.get_results_path("plots"), filename)
        plt.savefig(filepath)
        plt.close()
        log.info(f"Confusion matrix plot saved to {filepath}")

        return filepath

    def save_experiments_summary(self, all_metrics: Dict[str, Dict[str, Any]], all_times: List[float]) -> str:
        """
        Save a summary of all experiments/runs.

        This method generates a comprehensive summary of all experiment runs, including
        average performance metrics (accuracy, precision, recall, F1 score) and model
        characteristics (parameter counts, training times). It also includes individual
        results for each run. The summary is saved as a text file in the experiment directory.

        :param all_metrics: Dictionary of run metrics, with run IDs as keys
        :type all_metrics: Dict[str, Dict[str, Any]]
        :param all_times: List of running times for each experiment
        :type all_times: List[float]
        :return: Path to the saved summary file
        :rtype: str
        """
        # Calculate means and std devs
        accuracies = [metrics.get('accuracy', 0) for metrics in all_metrics.values()]
        precisions = [metrics.get('precision', 0) for metrics in all_metrics.values()]
        recalls = [metrics.get('recall', 0) for metrics in all_metrics.values()]
        f1_scores = [metrics.get('f1_score', 0) for metrics in all_metrics.values()]

        param_counts = [metrics.get('total_params', 0) for metrics in all_metrics.values()]

        summary = f"EXPERIMENT SUMMARY: {self.experiment_name}\n" + "=" * 60 + "\n\n"
        summary += f"Dataset: {self.dataset_name}\n"
        summary += f"Timestamp: {self.timestamp}\n"
        summary += f"Number of Runs: {len(all_metrics)}\n\n"

        summary += "PERFORMANCE METRICS (mean ± std):\n" + "-" * 40 + "\n"
        summary += f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n"
        summary += f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}\n"
        summary += f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}\n"
        summary += f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n\n"

        summary += "MODEL CHARACTERISTICS (mean ± std):\n" + "-" * 40 + "\n"
        summary += f"Parameters: {np.mean(param_counts):.0f} ± {np.std(param_counts):.0f}\n"
        summary += f"Average Training Time: {np.mean(all_times):.2f} ± {np.std(all_times):.2f} seconds\n\n"

        summary += "INDIVIDUAL RUN RESULTS:\n" + "-" * 40 + "\n"
        for i, (run_id, metrics) in enumerate(all_metrics.items()):
            summary += f"Run {i} - Accuracy: {metrics.get('accuracy', 0):.4f}, "
            summary += f"Parameters: {metrics.get('total_params', 0):,}, "
            summary += f"Time: {all_times[i]:.2f}s\n"

        filepath = os.path.join(self.experiment_dir, "experiment_summary.txt")
        with open(filepath, "w") as f:
            f.write(summary)

        log.info(f"Experiment summary saved to {filepath}")
        return filepath

    def save_architecture_as_json(self, architecture_str, filename, run_idx=None):
        """
        Save model architecture as a formatted JSON structure.

        This method parses a string representation of a model architecture and converts it
        into a structured JSON format. It extracts information about each layer, including
        layer type, kernel sizes, output channels, etc. The JSON structure is saved to a file
        in the metrics directory.

        :param architecture_str: String representation of the model architecture
        :type architecture_str: str
        :param filename: Name for the saved file
        :type filename: str
        :param run_idx: Run index to include in the filename
        :type run_idx: int or None
        :return: Path to the saved JSON file
        :rtype: str
        """
        import json
        import os

        # Parse the architecture string to extract layer types
        layers = []
        for layer in architecture_str.strip().split('|'):
            layer_type = layer.strip()
            if layer_type:
                layers.append(layer_type)

        # Create a JSON structure for the architecture
        architecture_json = {}

        # Add model information
        architecture_json["model"] = {
            "name": filename,
            "total_layers": len(layers)
        }

        # Add layers information
        architecture_json["layers"] = []

        # Process each layer
        for i, layer in enumerate(layers):
            layer_info = {
                "index": i,
                "type": layer
            }

            # Add specific information based on layer type
            if "conv" in layer:
                # For convolution layers, try to extract kernel and output channels
                layer_info["category"] = "Convolution"
                if "kernel" in layer:
                    try:
                        kernel_size = int(''.join(filter(str.isdigit, layer.split("kernel")[1].strip())))
                        layer_info["kernel_size"] = kernel_size
                    except:
                        pass
                if "ou_c" in layer:
                    try:
                        output_channels = int(''.join(filter(str.isdigit, layer.split("ou_c")[1].strip())))
                        layer_info["output_channels"] = output_channels
                    except:
                        pass

            elif "pool" in layer:
                # For pooling layers
                if "max_pool" in layer:
                    layer_info["category"] = "Max Pooling"
                elif "avg_pool" in layer:
                    layer_info["category"] = "Average Pooling"

            elif "fc" in layer:
                # For fully connected layers
                layer_info["category"] = "Fully Connected"
                if "ou_c" in layer:
                    try:
                        output_neurons = int(''.join(filter(str.isdigit, layer.split("ou_c")[1].strip())))
                        layer_info["output_neurons"] = output_neurons
                    except:
                        pass

            # Add layer to the architecture
            architecture_json["layers"].append(layer_info)

        # Count layer types
        layer_counts = {
            "convolution_layers": sum(1 for layer in layers if "conv" in layer),
            "max_pooling_layers": sum(1 for layer in layers if "max_pool" in layer),
            "avg_pooling_layers": sum(1 for layer in layers if "avg_pool" in layer),
            "fully_connected_layers": sum(1 for layer in layers if "fc" in layer)
        }

        architecture_json["summary"] = layer_counts

        # Determine the save path
        if run_idx is not None:
            save_path = os.path.join(self.visualizations_dir, f"{filename}_run{run_idx}.json")
        else:
            save_path = os.path.join(self.visualizations_dir, f"{filename}.json")

        # Save the architecture as a JSON file
        with open(save_path, 'w') as f:
            json.dump(architecture_json, f, indent=2)

        # Also save as a text file for better readability
        txt_path = save_path.replace('.json', '.txt')
        with open(txt_path, 'w') as f:
            f.write(f"MODEL ARCHITECTURE: {filename}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total layers: {len(layers)}\n\n")
            f.write("Layer structure:\n")

            for i, layer in enumerate(layers):
                f.write(f"  [{i}] {layer}\n")

            f.write("\nLayer counts:\n")
            for layer_type, count in layer_counts.items():
                f.write(f"  {layer_type}: {count}\n")

        print(f"Saved architecture JSON to: {save_path}")
        print(f"Saved architecture text to: {txt_path}")

        return save_path
