import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import argparse
import time
from datetime import datetime
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Set up consistent styling for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")
sns.set_context("talk")

# Custom color scheme
colors = ["#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c",
          "#f9d057", "#f29e2e", "#e76818", "#d7191c"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)


class PSO_MetricsVisualizer:
    """
    Creates comprehensive visualizations for PSO-CNN experiment results
    """

    def __init__(self, results_dir=None):
        """
        Initialize the visualizer for a specific results directory

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create visualization directory
        if results_dir:
            self.viz_dir = os.path.join(results_dir, "visualizations")
        else:
            self.viz_dir = f"./visualizations_{self.timestamp}"

        os.makedirs(self.viz_dir, exist_ok=True)

        print(f"Visualizations will be saved to: {self.viz_dir}")

    def find_all_result_directories(self):
        """Find all result directories in the ./results folder"""
        if os.path.exists("./results"):
            return [d for d in glob.glob("./results/*") if os.path.isdir(d)]
        return []

    def load_metrics_from_dir(self, directory):
        """Load all available metrics from a directory"""
        metrics = {}

        # Look for detailed metrics files
        for metrics_file in glob.glob(f"{directory}/metrics/*.npy"):
            try:
                data = np.load(metrics_file, allow_pickle=True).item()
                name = os.path.basename(metrics_file).replace(".npy", "")
                metrics[name] = data
            except:
                print(f"Could not load metrics from {metrics_file}")

        # Look for accuracy history files
        acc_history_files = glob.glob(f"{directory}/*acc_history.npy")
        for acc_file in acc_history_files:
            try:
                data = np.load(acc_file)
                name = os.path.basename(acc_file).replace(".npy", "")
                metrics[name] = data
            except:
                print(f"Could not load accuracy history from {acc_file}")

        return metrics

    def plot_accuracy_comparison(self, all_results, save=True):
        """Plot accuracy comparison across datasets/experiments"""
        datasets = []
        accuracies = []
        errors = []

        for dataset, results in all_results.items():
            if 'all_comprehensive_metrics' in results:
                # Extract accuracies from all runs
                accs = []
                for run_id, metrics in results['all_comprehensive_metrics'].items():
                    if 'accuracy' in metrics:
                        accs.append(metrics['accuracy'])

                if accs:
                    datasets.append(dataset)
                    accuracies.append(np.mean(accs))
                    errors.append(np.std(accs))

        if not datasets:
            print("No accuracy data found for comparison")
            return

        plt.figure(figsize=(12, 6))
        x = np.arange(len(datasets))
        bars = plt.bar(x, accuracies, yerr=errors, capsize=10,
                       color=sns.color_palette("muted", len(datasets)))

        plt.xlabel('Dataset')
        plt.ylabel('Test Accuracy')
        plt.title('PSO-CNN Performance Across Datasets')
        plt.xticks(x, datasets, rotation=45, ha='right')
        plt.tight_layout()

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom')

        if save:
            plt.savefig(os.path.join(self.viz_dir, "dataset_accuracy_comparison.png"))
            print(f"Saved accuracy comparison to {self.viz_dir}/dataset_accuracy_comparison.png")

        plt.show()

    def plot_metric_history(self, history, metric_name, title, save=True):
        """Plot a metric's history over iterations/epochs"""
        plt.figure(figsize=(10, 6))

        if isinstance(history, list):
            values = history
            x = range(1, len(values) + 1)
        elif isinstance(history, dict):
            x = list(history.keys())
            values = list(history.values())
        else:
            values = history
            x = range(1, len(values) + 1)

        plt.plot(x, values, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel(metric_name.capitalize())
        plt.title(title)

        for i, val in enumerate(values):
            plt.text(x[i], val + 0.01, f'{val:.4f}', ha='center')

        if save:
            filename = f"{metric_name}_history.png"
            plt.savefig(os.path.join(self.viz_dir, filename))
            print(f"Saved {metric_name} history to {self.viz_dir}/{filename}")

        plt.show()

    def plot_confusion_matrix(self, cm, classes, title="Confusion Matrix", normalize=False, save=True):
        """Plot a confusion matrix with proper labels and styling"""
        plt.figure(figsize=(10, 8))

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        sns.heatmap(cm, annot=True, fmt=fmt, cmap=custom_cmap,
                    xticklabels=classes, yticklabels=classes, cbar=True)

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.viz_dir, "confusion_matrix.png"))
            print(f"Saved confusion matrix to {self.viz_dir}/confusion_matrix.png")

        plt.show()

    def plot_architecture_comparison(self, all_results, save=True):
        """Plot model size vs accuracy for different architectures"""
        sizes = []
        accuracies = []
        params = []
        labels = []

        for dataset, results in all_results.items():
            if 'all_comprehensive_metrics' in results:
                for run_id, metrics in results['all_comprehensive_metrics'].items():
                    if 'accuracy' in metrics and 'total_params' in metrics:
                        sizes.append(metrics.get('model_size_mb', 0))
                        accuracies.append(metrics['accuracy'])
                        params.append(metrics['total_params'])
                        labels.append(f"{dataset}-{run_id}")

        if not sizes:
            print("No architecture metrics found for comparison")
            return

        # Plot size vs accuracy
        plt.figure(figsize=(12, 8))

        # Calculate size of scatter points based on parameter count
        norm_params = np.array(params) / max(params) * 500

        scatter = plt.scatter(sizes, accuracies, s=norm_params, alpha=0.6,
                              c=range(len(sizes)), cmap=custom_cmap)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Test Accuracy')
        plt.title('PSO-CNN Architecture Comparison')

        # Add annotations for points
        for i, label in enumerate(labels):
            plt.annotate(label, (sizes[i], accuracies[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

        # Add colorbar legend
        cbar = plt.colorbar(scatter)
        cbar.set_label('Architecture Index')

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.viz_dir, "architecture_comparison.png"))
            print(f"Saved architecture comparison to {self.viz_dir}/architecture_comparison.png")

        plt.show()

        # Plot parameters vs accuracy
        plt.figure(figsize=(12, 8))

        # Use log scale for parameter count
        scatter = plt.scatter(params, accuracies, s=200, alpha=0.6,
                              c=range(len(sizes)), cmap=custom_cmap)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Parameter Count')
        plt.ylabel('Test Accuracy')
        plt.title('PSO-CNN Parameters vs. Accuracy')
        plt.xscale('log')

        # Add annotations for points
        for i, label in enumerate(labels):
            plt.annotate(label, (params[i], accuracies[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

        # Add colorbar legend
        cbar = plt.colorbar(scatter)
        cbar.set_label('Architecture Index')

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.viz_dir, "params_vs_accuracy.png"))
            print(f"Saved params vs accuracy plot to {self.viz_dir}/params_vs_accuracy.png")

        plt.show()

    def plot_pso_convergence(self, all_results, save=True):
        """Plot PSO convergence across iterations for different datasets"""
        plt.figure(figsize=(12, 8))

        for dataset, results in all_results.items():
            # Find accuracy history files
            acc_history = None

            for key, value in results.items():
                if 'gBest_acc_history' in key and isinstance(value, np.ndarray):
                    acc_history = value
                    break

            if acc_history is not None:
                iterations = range(1, len(acc_history) + 1)
                plt.plot(iterations, acc_history, marker='o', linestyle='-',
                         linewidth=2, markersize=8, label=dataset)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Best Accuracy')
        plt.title('PSO Convergence Across Datasets')
        plt.legend(loc='lower right')
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.viz_dir, "pso_convergence.png"))
            print(f"Saved PSO convergence plot to {self.viz_dir}/pso_convergence.png")

        plt.show()

    def create_comprehensive_report(self):
        """Generate a comprehensive visual report of all available metrics"""
        result_dirs = self.find_all_result_directories()

        if not result_dirs:
            print("No result directories found in ./results/")
            return

        # Load all available metrics
        all_results = {}
        for result_dir in result_dirs:
            dataset_name = os.path.basename(result_dir)
            all_results[dataset_name] = self.load_metrics_from_dir(result_dir)

        # Create various plots
        print("Generating accuracy comparison across datasets...")
        self.plot_accuracy_comparison(all_results)

        print("Generating architecture comparison...")
        self.plot_architecture_comparison(all_results)

        print("Generating PSO convergence plots...")
        self.plot_pso_convergence(all_results)

        # Generate individual dataset metrics
        for dataset, results in all_results.items():
            print(f"\nProcessing dataset: {dataset}")

            # Set specific visualization dir for this dataset
            dataset_viz_dir = os.path.join(self.viz_dir, dataset)
            os.makedirs(dataset_viz_dir, exist_ok=True)
            original_viz_dir = self.viz_dir
            self.viz_dir = dataset_viz_dir

            # Plot accuracy history if available
            for key, value in results.items():
                if 'gBest_acc_history' in key and isinstance(value, np.ndarray):
                    self.plot_metric_history(
                        value,
                        'accuracy',
                        f'{dataset} - gBest Accuracy History'
                    )

                # Plot confusion matrix if available
                if 'all_comprehensive_metrics' in key:
                    for run_id, metrics in value.items():
                        if 'confusion_matrix' in metrics:
                            cm = metrics['confusion_matrix']
                            # Create generic class names if not available
                            class_names = [str(i) for i in range(cm.shape[0])]
                            self.plot_confusion_matrix(
                                cm,
                                class_names,
                                f'{dataset} - {run_id} Confusion Matrix'
                            )

            # Restore original visualization directory
            self.viz_dir = original_viz_dir

        # Create summary markdown file
        self.create_summary_markdown(all_results)

        print(f"\nComprehensive report generated in {self.viz_dir}")

    def create_summary_markdown(self, all_results):
        """Create a summary markdown file with links to all visualizations"""
        summary = f"# PSO-CNN Results Summary\n\n"
        summary += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Overall statistics
        summary += "## Overall Performance\n\n"
        summary += "| Dataset | Accuracy | Parameters | Model Size (MB) | Inference Time (ms) |\n"
        summary += "|---------|----------|------------|-----------------|---------------------|\n"

        for dataset, results in all_results.items():
            if 'all_comprehensive_metrics' in results:
                # Calculate averages across runs
                acc_sum = param_sum = size_sum = inf_sum = 0
                count = 0

                for run_id, metrics in results['all_comprehensive_metrics'].items():
                    if 'accuracy' in metrics:
                        acc_sum += metrics.get('accuracy', 0)
                        param_sum += metrics.get('total_params', 0)
                        size_sum += metrics.get('model_size_mb', 0)
                        inf_sum += metrics.get('inference_time_per_sample', 0) * 1000  # Convert to ms
                        count += 1

                if count > 0:
                    avg_acc = acc_sum / count
                    avg_params = param_sum / count
                    avg_size = size_sum / count
                    avg_inf = inf_sum / count

                    summary += f"| {dataset} | {avg_acc:.4f} | {avg_params:,.0f} | {avg_size:.2f} | {avg_inf:.2f} |\n"

        # Add links to visualizations
        summary += "\n## Visualizations\n\n"
        summary += "### Overall Comparisons\n\n"
        summary += "- [Dataset Accuracy Comparison](./dataset_accuracy_comparison.png)\n"
        summary += "- [Architecture Comparison](./architecture_comparison.png)\n"
        summary += "- [Parameters vs Accuracy](./params_vs_accuracy.png)\n"
        summary += "- [PSO Convergence](./pso_convergence.png)\n\n"

        # Add dataset-specific visualizations
        summary += "### Dataset-Specific Visualizations\n\n"

        for dataset in all_results.keys():
            summary += f"#### {dataset}\n\n"
            summary += f"- [Accuracy History](./{dataset}/accuracy_history.png)\n"
            summary += f"- [Confusion Matrix](./{dataset}/confusion_matrix.png)\n\n"

        # Write summary to file
        with open(os.path.join(self.viz_dir, "summary.md"), "w") as f:
            f.write(summary)

        print(f"Created summary markdown file: {self.viz_dir}/summary.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comprehensive metrics visualization for PSO-CNN experiments")
    parser.add_argument("--results_dir", type=str, help="Specific results directory to visualize")
    args = parser.parse_args()

    visualizer = PSO_MetricsVisualizer(args.results_dir)
    visualizer.create_comprehensive_report()