"""
Main script for running PSO-CNN experiments.

This script configures and executes Particle Swarm Optimization (PSO) for finding optimal
CNN architectures for image classification tasks. It handles the complete workflow:
1. Setting up experiment parameters
2. Running PSO to find the best CNN architecture
3. Training the best model found
4. Evaluating performance
5. Saving results, metrics, and visualizations

The script supports multiple runs of the algorithm and collects comprehensive metrics
for analysis and comparison.
"""

import gc
import time

import matplotlib
import matplotlib.pyplot as plt

from metrics import ResultsManager
from psoCNN import psoCNN

if __name__ == '__main__':
    ######## Algorithm parameters ##################

    # dataset = "mnist"
    # dataset = "mnist-rotated-digits"
    # dataset = "mnist-rotated-with-background"
    # dataset = "rectangles"
    # dataset = "rectangles-images"
    # dataset = "convex"
    # dataset = "fashion-mnist"
    # dataset = "mnist-random-background"
    # dataset = "mnist-background-images"
    dataset = "early-dataset"
    # dataset = "late-color-dataset"

    number_runs = 2
    number_iterations = 5
    population_size = 5

    batch_size_pso = 32
    batch_size_full_training = 32

    epochs_pso = 2
    epochs_full_training = 50

    max_conv_output_channels = 64
    max_fully_connected_neurons = 32

    min_layer = 3
    max_layer = 6

    # Probability of each layer type (should sum to 1)
    probability_convolution = 0.6
    probability_pooling = 0.2
    probability_fully_connected = 0.2

    max_conv_kernel_size = 5

    Cg = 0.4
    dropout = 0.4

    ########### Run the algorithm ######################
    results_manager = ResultsManager(dataset_name=dataset)

    # Dictionary to store comprehensive metrics for each run
    all_metrics = {}
    all_times = []
    best_gBest_acc = 0

    for i in range(number_runs):
        print(f"Run number: {i}")
        start_time = time.time()

        # Initialize and run PSO-CNN
        pso = psoCNN(dataset=dataset, n_iter=number_iterations, pop_size=population_size, batch_size=batch_size_pso,
                     epochs=epochs_pso, min_layer=min_layer, max_layer=max_layer, conv_prob=probability_convolution,
                     pool_prob=probability_pooling, fc_prob=probability_fully_connected,
                     max_conv_kernel=max_conv_kernel_size, max_out_ch=max_conv_output_channels,
                     max_fc_neurons=max_fully_connected_neurons, dropout_rate=dropout)

        pso.fit(Cg=Cg, dropout_rate=dropout)

        print(pso.gBest_acc)

        # Plot and save gBest accuracy history
        results_manager.save_plot(data=pso.gBest_acc, filename=f"gBest_accuracy", ylabel="gBest Accuracy", run_idx=i)

        # Save gBest architecture as text
        architecture_text = f'gBest architecture:\n{pso.gBest}'
        results_manager.save_text_summary(text=architecture_text, filename="gBest_architecture", run_idx=i)

        # Save accuracy history arrays
        results_manager.save_metrics(metrics={'accuracy_history': pso.gBest_acc}, filename="gBest_acc_history",
            run_idx=i)

        results_manager.save_metrics(metrics={'test_accuracy_history': pso.gBest_test_acc},
            filename="gBest_test_acc_history", run_idx=i)

        # Record training time
        end_time = time.time()
        running_time = end_time - start_time
        all_times.append(running_time)

        # Fully train the gBest model found
        n_parameters = pso.fit_gBest(batch_size=batch_size_full_training, epochs=epochs_full_training,
                                     dropout_rate=dropout)

        # Evaluate the fully trained gBest model
        gBest_metrics = pso.evaluate_gBest(batch_size=batch_size_full_training)

        if gBest_metrics[1] >= best_gBest_acc:
            best_gBest_acc = gBest_metrics[1]

            # Save best gBest model
            results_manager.save_model(model=pso.gBest.model, name="best_gBest_model")

            # Evaluate comprehensive metrics
            detailed_metrics, metrics_summary = results_manager.evaluate_model(model=pso.gBest.model, x_test=pso.x_test,
                y_test=pso.y_test, batch_size=batch_size_full_training, training_time=running_time)

            # Store the metrics for this run
            all_metrics[f"run_{i}"] = detailed_metrics

            # Print and save detailed metrics summary
            print(metrics_summary)
            results_manager.save_text_summary(text=metrics_summary, filename="detailed_metrics")

            # Plot confusion matrix if it exists in metrics
            if 'confusion_matrix' in detailed_metrics:
                results_manager.plot_confusion_matrix(cm=detailed_metrics['confusion_matrix'], run_idx=i)
            del detailed_metrics
            del metrics_summary

        # Save all metrics for this run
        run_summary = {'loss': gBest_metrics[0], 'accuracy': gBest_metrics[1], 'parameters': n_parameters,
            'training_time': running_time}
        results_manager.save_metrics(metrics=run_summary, filename="run_metrics", run_idx=i)

        print(f"Run {i} took: {running_time:.2f} seconds.")

        # Free up memory
        del pso
        gc.collect()
        matplotlib.pyplot.close('all')

    # Save final summary of all runs
    results_manager.save_experiments_summary(all_metrics=all_metrics, all_times=all_times)

    # Also save the comprehensive metrics collection
    results_manager.save_metrics(metrics=all_metrics, filename="all_comprehensive_metrics")

    print(f"Experiment complete. Results saved to: {results_manager.experiment_dir}")
