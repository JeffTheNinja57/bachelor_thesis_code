import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from tensorflow.keras import backend as K
from metrics import ResultsManager

from psoCNN import psoCNN


class LateFusionVisualizer:
    """
    A class for visualizing training results and model architecture for late fusion experiments.

    It organizes results into a directory structure based on experiment name and timestamp.
    It provides methods for saving training history, confusion matrices, model comparisons,
    and model architecture visualizations.
    """
    def __init__(self, base_dir="results", experiment_name="late_fusion_experiment"):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.base_dir = os.path.join(base_dir, f"{self.experiment_name}_{self.timestamp}")

        self.color_dir = os.path.join(self.base_dir, "color")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.fusion_dir = os.path.join(self.base_dir, "fusion")

        self.color_viz_dir = os.path.join(self.color_dir, "visualizations")
        self.depth_viz_dir = os.path.join(self.depth_dir, "visualizations")
        self.fusion_viz_dir = os.path.join(self.fusion_dir, "visualizations")

        os.makedirs(self.color_viz_dir, exist_ok=True)
        os.makedirs(self.depth_viz_dir, exist_ok=True)
        os.makedirs(self.fusion_viz_dir, exist_ok=True)

        print(f"Experiment setup complete. Results will be saved to: {self.base_dir}")

    def save_training_history(self, history, model_name, viz_dir):
        """Saves the training history to a PNG image.

        This function visualizes the accuracy and loss during training
        and saves the plot to a PNG file in the specified directory.

        :param history: Training history object.
        :type history:
        :param model_name: Name of the model.
        :type model_name: str
        :param viz_dir: Directory to save the visualization.
        :type viz_dir: str
        :return: None
        :rtype: None
        """
        if history is None or not hasattr(history, 'history'):
            print(f"No history to save for {model_name}.")
            return

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        history_plot_path = os.path.join(viz_dir, f"{model_name}_training_history.png")
        plt.savefig(history_plot_path)
        plt.close()
        print(f"Saved {model_name} training history to: {history_plot_path}")

    def plot_confusion_matrix(self, y_true, y_pred_classes, class_names, model_name, viz_dir):
        """Plots and saves a confusion matrix for given predictions.

        The confusion matrix visualizes the performance of a classification
        model by comparing predicted classes to true classes. It's saved
        as an image file in the specified visualization directory.

        :param y_true: The true class labels.
        :type y_true: :class:`numpy.ndarray`
        :param y_pred_classes: The predicted class labels.
        :type y_pred_classes: :class:`numpy.ndarray`
        :param class_names: A list of class names corresponding to the labels.
        :type class_names: list of :class:`str`
        :param model_name: The name of the model being evaluated.
        :type model_name: :class:`str`
        :param viz_dir: The directory where the confusion matrix image will be saved.
        :type viz_dir: :class:`str`
        :return: None
        :rtype: None
        """
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_path = os.path.join(viz_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved {model_name} confusion matrix to: {cm_path}")

    def plot_model_comparison(self, metrics_dict, output_path):
        """
        Plots a bar chart comparing the performance of different models.

        The chart displays accuracy, precision, recall, and F1-score for each model
        using a grouped bar chart format.  The plot is saved as a PNG image
        to the specified output path.

        :param metrics_dict: A dictionary where keys are model names
                               and values are dictionaries containing performance metrics.
                               Each inner dictionary should contain 'accuracy', 'precision',
                               'recall', and 'f1_score' keys.
        :type metrics_dict: dict
        :param output_path: The file path to save the generated plot.
        :type output_path: str
        :return: None
        :rtype: None
        """
        # metrics_dict = {'ModelName': {'accuracy': 0.9, 'precision': 0.8, ...}, ...}
        labels = list(metrics_dict.keys())
        accuracy_scores = [m.get('accuracy', 0) for m in metrics_dict.values()]
        precision_scores = [m.get('precision', 0) for m in metrics_dict.values()]  # Assuming macro precision
        recall_scores = [m.get('recall', 0) for m in metrics_dict.values()]  # Assuming macro recall
        f1_scores = [m.get('f1_score', 0) for m in metrics_dict.values()]  # Assuming macro f1

        x = np.arange(len(labels))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 7))
        rects1 = ax.bar(x - 1.5 * width, accuracy_scores, width, label='Accuracy')
        rects2 = ax.bar(x - 0.5 * width, precision_scores, width, label='Precision (Macro)')
        rects3 = ax.bar(x + 0.5 * width, recall_scores, width, label='Recall (Macro)')
        rects4 = ax.bar(x + 1.5 * width, f1_scores, width, label='F1-score (Macro)')

        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.bar_label(rects1, padding=3, fmt='%.2f')
        ax.bar_label(rects2, padding=3, fmt='%.2f')
        ax.bar_label(rects3, padding=3, fmt='%.2f')
        ax.bar_label(rects4, padding=3, fmt='%.2f')

        fig.tight_layout()
        plt.savefig(os.path.join(self.base_dir, "model_comparison_summary.png"))
        plt.close()
        print(f"Saved model comparison plot to: {os.path.join(self.base_dir, 'model_comparison_summary.png')}")

    def visualize_architecture(self, model, model_name, viz_dir):
        """Visualizes the architecture of a Keras model.

        This function generates a visual representation and a text summary
        of the given Keras model.  The visual representation is saved as
        a PNG file, and the text summary is saved as a text file.
        If the model is None, a message is printed and the function returns.
        If an error occurs during the plotting or saving process,
        an error message is printed.

        :param model: The Keras model to visualize.
        :type model: tf.keras.Model
        :param model_name: The name of the model.
        :type model_name: str
        :param viz_dir: The directory to save the visualizations to.
        :type viz_dir: str
        :return: None
        :rtype: None
        """
        if model is None:
            print(f"No model to visualize for {model_name}.")
            return
        try:
            plot_path = os.path.join(viz_dir, f"{model_name}_architecture.png")
            tf.keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
            print(f"Saved {model_name} architecture plot to: {plot_path}")

            # Save architecture as text
            text_path = os.path.join(viz_dir, f"{model_name}_architecture.txt")
            with open(text_path, 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            print(f"Saved {model_name} architecture text to: {text_path}")

        except Exception as e:
            print(f"Could not plot/save {model_name} architecture: {e}")


class LateFusionPSOCNN:
    """
    A class for implementing late fusion of color and depth models using Particle Swarm Optimization CNN.

    This class handles the complete workflow of training separate CNN models for color and depth data,
    and then performing late fusion by combining their predictions. The CNN architectures are optimized
    using Particle Swarm Optimization (PSO) to find effective network structures for each modality.

    The class provides methods for training individual models, performing fusion, and evaluating results.
    It also handles visualization and metrics collection through the LateFusionVisualizer class.
    """
    def __init__(self, color_dataset, depth_dataset, number_iterations=5, population_size=5, batch_size_pso=32,
                 epochs_pso=2, min_layer=3, max_layer=8, probability_convolution=0.6, probability_pooling=0.2,
                 probability_fully_connected=0.2, max_conv_kernel_size=7, max_conv_output_channels=256,
                 max_fully_connected_neurons=300, dropout=0.4, Cg=0.5, epochs_full_training=10,
                 batch_size_full_training=32, number_runs=1):
        """
        Initialize the LateFusionPSOCNN with datasets and parameters.

        :param color_dataset: Name or path of the color dataset
        :type color_dataset: str
        :param depth_dataset: Name or path of the depth dataset
        :type depth_dataset: str
        :param number_iterations: Number of PSO iterations
        :type number_iterations: int
        :param population_size: Size of the PSO population
        :type population_size: int
        :param batch_size_pso: Batch size for PSO training
        :type batch_size_pso: int
        :param epochs_pso: Number of epochs for PSO training
        :type epochs_pso: int
        :param min_layer: Minimum number of layers in CNN architecture
        :type min_layer: int
        :param max_layer: Maximum number of layers in CNN architecture
        :type max_layer: int
        :param probability_convolution: Probability of adding a convolutional layer
        :type probability_convolution: float
        :param probability_pooling: Probability of adding a pooling layer
        :type probability_pooling: float
        :param probability_fully_connected: Probability of adding a fully connected layer
        :type probability_fully_connected: float
        :param max_conv_kernel_size: Maximum convolutional kernel size
        :type max_conv_kernel_size: int
        :param max_conv_output_channels: Maximum number of output channels for convolutional layers
        :type max_conv_output_channels: int
        :param max_fully_connected_neurons: Maximum number of neurons in fully connected layers
        :type max_fully_connected_neurons: int
        :param dropout: Dropout rate for regularization
        :type dropout: float
        :param Cg: PSO global best weight parameter
        :type Cg: float
        :param epochs_full_training: Number of epochs for full training after PSO
        :type epochs_full_training: int
        :param batch_size_full_training: Batch size for full training
        :type batch_size_full_training: int
        :param number_runs: Number of runs to perform
        :type number_runs: int
        """
        self.color_dataset = color_dataset
        self.depth_dataset = depth_dataset
        self.visualizer = LateFusionVisualizer(experiment_name="late_fusion")

        # PSO and Model Parameters
        self.number_iterations = number_iterations
        self.population_size = population_size
        self.batch_size_pso = batch_size_pso
        self.epochs_pso = epochs_pso
        self.min_layer = min_layer
        self.max_layer = max_layer
        self.probability_convolution = probability_convolution
        self.probability_pooling = probability_pooling
        self.probability_fully_connected = probability_fully_connected
        self.max_conv_kernel_size = max_conv_kernel_size
        self.max_conv_output_channels = max_conv_output_channels
        self.max_fully_connected_neurons = max_fully_connected_neurons
        self.dropout = dropout
        self.Cg = Cg  # PSO specific parameter

        # Full Training Parameters
        self.epochs_full_training = epochs_full_training
        self.batch_size_full_training = batch_size_full_training
        self.number_runs = number_runs

        # Placeholders for models and data
        self.color_model = None
        self.depth_model = None
        self.fusion_model = None
        self.color_x_test, self.color_y_test = None, None
        self.depth_x_test, self.depth_y_test = None, None

        # Metrics storage
        self.color_metrics = {}
        self.depth_metrics = {}
        self.fusion_metrics = {}

    def _fix_data_cardinality(self, x1_test, y1_test, x2_test, y2_test):
        """
        Ensure both test datasets have the same number of samples by truncating the larger one.

        This method is used to make sure that both color and depth test datasets have the same
        number of samples, which is necessary for late fusion. It truncates both datasets to
        the size of the smaller one and performs a sanity check to verify that the labels are
        aligned between the two datasets.

        :param x1_test: First dataset features (typically color)
        :type x1_test: numpy.ndarray
        :param y1_test: First dataset labels
        :type y1_test: numpy.ndarray
        :param x2_test: Second dataset features (typically depth)
        :type x2_test: numpy.ndarray
        :param y2_test: Second dataset labels
        :type y2_test: numpy.ndarray
        :return: Truncated datasets (x1_test_fixed, y1_test_fixed, x2_test_fixed, y2_test_fixed)
        :rtype: tuple of numpy.ndarray
        """
        min_samples = min(len(x1_test), len(x2_test))

        x1_test_fixed = x1_test[:min_samples]
        y1_test_fixed = y1_test[:min_samples]
        x2_test_fixed = x2_test[:min_samples]
        y2_test_fixed = y2_test[:min_samples]

        print(f"Data cardinality fixed: Using {min_samples} samples from each test set.")
        print(f"Original x1_test: {len(x1_test)}, y1_test: {len(y1_test)}")
        print(f"Original x2_test: {len(x2_test)}, y2_test: {len(y2_test)}")
        print(f"Fixed x1_test: {len(x1_test_fixed)}, y1_test: {len(y1_test_fixed)}")
        print(f"Fixed x2_test: {len(x2_test_fixed)}, y2_test: {len(y2_test_fixed)}")

        # Sanity check: Ensure y_test for color and depth are the same after fixing cardinality if they come from aligned sources
        # This depends on how datasets were created. If they are truly aligned (same samples, different modalities)
        # then y1_test_fixed should be identical to y2_test_fixed.
        if not np.array_equal(np.argmax(y1_test_fixed, axis=1), np.argmax(y2_test_fixed, axis=1)):
            print(
                "Warning: y_test for color and depth models are not identical after cardinality fix. Ensure datasets are aligned.")

        return x1_test_fixed, y1_test_fixed, x2_test_fixed, y2_test_fixed

    def train_color_model(self):
        """
        Train a CNN model on the color dataset using PSO for architecture optimization.

        This method initializes a PSO-CNN for the color dataset, runs the PSO algorithm to find
        an optimal architecture, and then fully trains the best model found. It also evaluates
        the model on the test set, calculates performance metrics, and saves visualizations.

        The trained model is stored in self.color_model, and the test data is stored in
        self.color_x_test and self.color_y_test. Performance metrics are stored in self.color_metrics.

        :return: None
        :rtype: None
        """
        print("\n" + "=" * 50)
        print(f"TRAINING COLOR MODEL USING DATASET: {self.color_dataset}")
        print("=" * 50 + "\n")

        self.color_pso = psoCNN(dataset=self.color_dataset, n_iter=self.number_iterations,
                                pop_size=self.population_size, batch_size=self.batch_size_pso, epochs=self.epochs_pso,
                                min_layer=self.min_layer, max_layer=self.max_layer,
                                conv_prob=self.probability_convolution, pool_prob=self.probability_pooling,
                                fc_prob=self.probability_fully_connected, max_conv_kernel=self.max_conv_kernel_size,
                                max_out_ch=self.max_conv_output_channels,
                                max_fc_neurons=self.max_fully_connected_neurons, dropout_rate=self.dropout,
                                activation='relu')
        self.color_pso.fit(Cg=self.Cg, dropout_rate=self.dropout)
        print(
            f"COLOR MODEL BEST PSO ACCURACY: {self.color_pso.gBest_acc[-1] if len(self.color_pso.gBest_acc) > 0 else 'N/A'}")

        n_params_color = self.color_pso.fit_gBest(batch_size=self.batch_size_full_training,
                                                  epochs=self.epochs_full_training, dropout_rate=self.dropout)
        self.color_model = self.color_pso.gBest.model
        self.color_x_test, self.color_y_test = self.color_pso.x_test, self.color_pso.y_test

        # Save and visualize
        if self.color_model:
            model_save_path = os.path.join(self.visualizer.color_dir, "color_model_final.keras")
            self.color_model.save(model_save_path)
            print(f"Color model saved to {model_save_path}")
            self.visualizer.visualize_architecture(self.color_model, "color_model",
                                                   self.visualizer.color_viz_dir)  # self.visualizer.save_training_history(color_pso.gBest.history, "color_model", self.visualizer.color_viz_dir) # History might be complex with PSO

        # Evaluate
        if self.color_model and self.color_x_test is not None and self.color_y_test is not None:
            y_pred_color_probs = self.color_model.predict(self.color_x_test)
            y_pred_color_classes = np.argmax(y_pred_color_probs, axis=1)
            y_true_color_classes = np.argmax(self.color_y_test, axis=1)

            self.color_metrics['accuracy'] = accuracy_score(y_true_color_classes, y_pred_color_classes)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true_color_classes, y_pred_color_classes,
                                                                       average='macro', zero_division=0)
            self.color_metrics['precision'] = precision
            self.color_metrics['recall'] = recall
            self.color_metrics['f1_score'] = f1
            self.color_metrics['loss'] = self.color_model.evaluate(self.color_x_test, self.color_y_test, verbose=0)[0]
            self.color_metrics['parameters'] = n_params_color

            print("\nCOLOR MODEL METRICS")
            print("===================")
            print(classification_report(y_true_color_classes, y_pred_color_classes, zero_division=0))
            print(f"Accuracy: {self.color_metrics['accuracy']:.4f}")
            print(f"Loss: {self.color_metrics['loss']:.4f}")
            print(f"Parameters: {n_params_color}")

            # Assuming class names are just indices for now if not provided
            num_classes = self.color_y_test.shape[1]
            class_names = [str(i) for i in range(num_classes)]
            self.visualizer.plot_confusion_matrix(y_true_color_classes, y_pred_color_classes, class_names,
                                                  "color_model", self.visualizer.color_viz_dir)
        else:
            print("Color model training or data loading failed, skipping evaluation.")

    def train_depth_model(self):
        """
        Train a CNN model on the depth dataset using PSO for architecture optimization.

        This method initializes a PSO-CNN for the depth dataset, runs the PSO algorithm to find
        an optimal architecture, and then fully trains the best model found. It uses LeakyReLU
        activation functions which are often better suited for depth data. The method also evaluates
        the model on the test set, calculates performance metrics, and saves visualizations.

        The trained model is stored in self.depth_model, and the test data is stored in
        self.depth_x_test and self.depth_y_test. Performance metrics are stored in self.depth_metrics.

        :return: None
        :rtype: None
        """
        print("\n" + "=" * 50)
        print(f"TRAINING DEPTH MODEL USING DATASET: {self.depth_dataset}")
        print("USING LeakyReLU ACTIVATION FOR DEPTH MODEL")
        print("=" * 50 + "\n")

        self.depth_pso = psoCNN(dataset=self.depth_dataset, n_iter=self.number_iterations,
                                pop_size=self.population_size, batch_size=self.batch_size_pso, epochs=self.epochs_pso,
                                min_layer=self.min_layer, max_layer=self.max_layer,
                                conv_prob=self.probability_convolution, pool_prob=self.probability_pooling,
                                fc_prob=self.probability_fully_connected, max_conv_kernel=self.max_conv_kernel_size,
                                max_out_ch=self.max_conv_output_channels,
                                max_fc_neurons=self.max_fully_connected_neurons, dropout_rate=self.dropout,
                                activation='leaky_relu'  # Use LeakyReLU activation for depth
                                )
        self.depth_pso.fit(Cg=self.Cg, dropout_rate=self.dropout)
        print(
            f"DEPTH MODEL BEST PSO ACCURACY: {self.depth_pso.gBest_acc[-1] if len(self.depth_pso.gBest_acc) > 0 else 'N/A'}")

        n_params_depth = self.depth_pso.fit_gBest(batch_size=self.batch_size_full_training,
                                                  epochs=self.epochs_full_training, dropout_rate=self.dropout)
        self.depth_model = self.depth_pso.gBest.model
        self.depth_x_test, self.depth_y_test = self.depth_pso.x_test, self.depth_pso.y_test

        # Save and visualize
        if self.depth_model:
            model_save_path = os.path.join(self.visualizer.depth_dir, "depth_model_final.keras")
            self.depth_model.save(model_save_path)
            print(f"Depth model saved to {model_save_path}")
            self.visualizer.visualize_architecture(self.depth_model, "depth_model",
                                                   self.visualizer.depth_viz_dir)  # self.visualizer.save_training_history(depth_pso.gBest.history, "depth_model", self.visualizer.depth_viz_dir)

        # Evaluate
        if self.depth_model and self.depth_x_test is not None and self.depth_y_test is not None:
            y_pred_depth_probs = self.depth_model.predict(self.depth_x_test)
            y_pred_depth_classes = np.argmax(y_pred_depth_probs, axis=1)
            y_true_depth_classes = np.argmax(self.depth_y_test, axis=1)

            self.depth_metrics['accuracy'] = accuracy_score(y_true_depth_classes, y_pred_depth_classes)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true_depth_classes, y_pred_depth_classes,
                                                                       average='macro', zero_division=0)
            self.depth_metrics['precision'] = precision
            self.depth_metrics['recall'] = recall
            self.depth_metrics['f1_score'] = f1
            self.depth_metrics['loss'] = self.depth_model.evaluate(self.depth_x_test, self.depth_y_test, verbose=0)[0]
            self.depth_metrics['parameters'] = n_params_depth

            print("\nDEPTH MODEL METRICS")
            print("===================")
            print(classification_report(y_true_depth_classes, y_pred_depth_classes, zero_division=0))
            print(f"Accuracy: {self.depth_metrics['accuracy']:.4f}")
            print(f"Loss: {self.depth_metrics['loss']:.4f}")
            print(f"Parameters: {n_params_depth}")

            num_classes = self.depth_y_test.shape[1]
            class_names = [str(i) for i in range(num_classes)]
            self.visualizer.plot_confusion_matrix(y_true_depth_classes, y_pred_depth_classes, class_names,
                                                  "depth_model", self.visualizer.depth_viz_dir)
        else:
            print("Depth model training or data loading failed, skipping evaluation.")

    def perform_late_fusion(self):
        """
        Perform late fusion by combining predictions from color and depth models.

        This method takes the trained color and depth models and combines their predictions
        on the test data. By default, it uses simple averaging of the probability outputs
        from both models, but other fusion strategies are available as commented options
        (weighted averaging, maximum probability, minimum probability).

        The method evaluates the fusion performance, calculates metrics, and generates
        visualizations. The fusion metrics are stored in self.fusion_metrics.

        :return: Fusion metrics dictionary
        :rtype: dict
        """
        print("\n" + "=" * 50)
        print("PERFORMING LATE FUSION")
        print("=" * 50 + "\n")

        if self.color_model is None or self.depth_model is None:
            print("One or both base models are not trained. Skipping fusion.")
            return None, {}

        # Ensure test data is loaded and has same number of samples
        if self.color_x_test is None or self.depth_x_test is None:
            print("Test data for one or both models not available. Cannot perform fusion.")
            return None, {}

        y_true_fusion_test = self.color_y_test

        # Get predictions (probabilities) from both models on their respective test sets
        color_predictions = self.color_model.predict(self.color_x_test)
        depth_predictions = self.depth_model.predict(self.depth_x_test)

        # Option 1: Simple averaging (default)
        fused_predictions_avg = (color_predictions + depth_predictions) / 2.0

        # Option 2: Weighted averaging (adjust weights based on model performance)
        # color_weight = 0.6
        # depth_weight = 0.4  # Make sure weights sum to 1.0
        # fused_predictions_avg = (color_weight * color_predictions + depth_weight * depth_predictions)

        # Option 3: Maximum probability (selects the class with highest confidence from either model)
        # fused_predictions_avg = np.maximum(color_predictions, depth_predictions)

        # Option 4: Minimum probability (more conservative approach)
        # fused_predictions_avg = np.minimum(color_predictions, depth_predictions)

        y_pred_fusion_classes_avg = np.argmax(fused_predictions_avg, axis=1)
        y_true_fusion_classes = np.argmax(y_true_fusion_test, axis=1)  # Use the common y_test

        self.fusion_metrics['late_avg'] = {}
        self.fusion_metrics['late_avg']['accuracy'] = accuracy_score(y_true_fusion_classes, y_pred_fusion_classes_avg)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_fusion_classes, y_pred_fusion_classes_avg,
                                                                   average='macro', zero_division=0)
        self.fusion_metrics['late_avg']['precision'] = precision
        self.fusion_metrics['late_avg']['recall'] = recall
        self.fusion_metrics['late_avg']['f1_score'] = f1
        cce = tf.keras.losses.CategoricalCrossentropy()
        self.fusion_metrics['late_avg']['loss'] = cce(y_true_fusion_test, fused_predictions_avg).numpy()

        print("\nLATE FUSION (Averaging Probabilities) METRICS")
        print("==============================================")
        print(classification_report(y_true_fusion_classes, y_pred_fusion_classes_avg, zero_division=0))
        print(f"Accuracy: {self.fusion_metrics['late_avg']['accuracy']:.4f}")
        print(f"Loss (CCE of avg_probs): {self.fusion_metrics['late_avg']['loss']:.4f}")

        num_classes = y_true_fusion_test.shape[1]
        class_names = [str(i) for i in range(num_classes)]
        self.visualizer.plot_confusion_matrix(y_true_fusion_classes, y_pred_fusion_classes_avg, class_names,
                                              "fusion_model_avg_probs", self.visualizer.fusion_viz_dir)

        return self.fusion_metrics

    def run_complete_workflow(self):
        """
        Run the complete late fusion workflow from training to evaluation.

        This method orchestrates the entire process:
        1. Trains the color model using PSO
        2. Trains the depth model using PSO
        3. Performs late fusion of the two models
        4. Compares the performance of all models
        5. Saves all metrics and visualizations

        It also tracks and reports timing information for each step and the overall process.
        All results are saved to the directories specified by the visualizer.

        :return: None
        :rtype: None
        """
        start_time_total = time.time()

        # Train Color Model
        color_train_start = time.time()
        self.train_color_model()
        self.color_metrics['training_time_seconds'] = time.time() - color_train_start
        # Store the gbest architecture
        if hasattr(self.color_pso, 'gBest') and self.color_pso.gBest is not None:
            self.color_metrics['gbest_architecture'] = str(self.color_pso.gBest)
            # Save the gbest architecture as JSON
            results_manager = ResultsManager(dataset_name=self.color_dataset)
            results_manager.save_architecture_as_json(str(self.color_pso.gBest), "color_gbest_architecture")

        # Train Depth Model
        depth_train_start = time.time()
        self.train_depth_model()
        self.depth_metrics['training_time_seconds'] = time.time() - depth_train_start
        # Store the gbest architecture
        if hasattr(self.depth_pso, 'gBest') and self.depth_pso.gBest is not None:
            self.depth_metrics['gbest_architecture'] = str(self.depth_pso.gBest)
            # Save the gbest architecture as JSON
            results_manager = ResultsManager(dataset_name=self.depth_dataset)
            results_manager.save_architecture_as_json(str(self.depth_pso.gBest), "depth_gbest_architecture")

        # Perform Late Fusion
        fusion_start = time.time()
        self.perform_late_fusion()
        if 'late_avg' in self.fusion_metrics: self.fusion_metrics['late_avg'][
            'processing_time_seconds'] = time.time() - fusion_start
        if 'late_mlp' in self.fusion_metrics: self.fusion_metrics['late_mlp'][
            'processing_time_seconds'] = time.time() - fusion_start  # Approx, includes MLP training if done

        total_time = time.time() - start_time_total
        print(f"\nTotal workflow completed in {total_time:.2f} seconds.")

        # Consolidate all metrics for comparison plot
        all_model_metrics = {}
        if self.color_metrics: all_model_metrics['Color'] = self.color_metrics
        if self.depth_metrics: all_model_metrics['Depth'] = self.depth_metrics
        if 'late_avg' in self.fusion_metrics and self.fusion_metrics['late_avg']:
            all_model_metrics['Fusion (Avg)'] = self.fusion_metrics['late_avg']
        if 'late_mlp' in self.fusion_metrics and self.fusion_metrics['late_mlp']:
            all_model_metrics['Fusion (MLP)'] = self.fusion_metrics['late_mlp']

        if all_model_metrics:
            self.visualizer.plot_model_comparison(all_model_metrics, self.visualizer.base_dir)

        # Save all collected metrics to a file
        metrics_summary_path = os.path.join(self.visualizer.base_dir, "all_metrics_summary.txt")
        with open(metrics_summary_path, 'w') as f:
            f.write("Color Model Metrics:\n")
            for k, v in self.color_metrics.items(): f.write(f"  {k}: {v}\n")
            f.write("\nDepth Model Metrics:\n")
            for k, v in self.depth_metrics.items(): f.write(f"  {k}: {v}\n")
            f.write("\nFusion Model Metrics:\n")
            for fusion_type, metrics_vals in self.fusion_metrics.items():
                f.write(f"  Type: {fusion_type}\n")
                for k, v in metrics_vals.items(): f.write(f"    {k}: {v}\n")
            f.write(f"\nTotal Workflow Time: {total_time:.2f} seconds\n")
        print(f"All metrics summary saved to: {metrics_summary_path}")


if __name__ == '__main__':
    # for i in range():
    late_fusion = LateFusionPSOCNN(color_dataset="late-color-dataset",
                                   depth_dataset="late-depth-dataset", number_iterations=5, population_size=5,
                                   epochs_pso=2, max_conv_kernel_size=5, max_conv_output_channels=64,
                                   max_fully_connected_neurons=32, probability_convolution=0.6, probability_pooling=0.2,
                                   probability_fully_connected=0.2, max_layer=4, epochs_full_training=25,
                                   batch_size_pso=64, batch_size_full_training=64)
    late_fusion.run_complete_workflow()
