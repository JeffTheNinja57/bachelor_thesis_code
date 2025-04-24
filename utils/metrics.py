import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import os
from thop import profile

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model):
    """
    Calculate the size of a model in MB.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        float: Model size in MB
    """
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6  # Convert bytes to MB
    os.remove("temp.p")
    return size

def calculate_flops(model, input_size, device):
    """
    Calculate the number of FLOPs for a model.
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input size (channels, height, width)
        device (torch.device): Device to run the calculation on
        
    Returns:
        float: Number of FLOPs
    """
    if isinstance(input_size[0], tuple):  # Late fusion
        # For late fusion models, we need to create two dummy inputs
        input1 = torch.randn(1, input_size[0][0], input_size[1], input_size[2]).to(device)
        input2 = torch.randn(1, input_size[0][1], input_size[1], input_size[2]).to(device)
        flops, _ = profile(model, inputs=(input1, input2))
    else:  # Early fusion
        input = torch.randn(1, input_size[0], input_size[1], input_size[2]).to(device)
        flops, _ = profile(model, inputs=(input,))
    
    return flops

def measure_inference_time(model, input_size, device, num_iterations=100):
    """
    Measure the inference time of a model.
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input size (channels, height, width)
        device (torch.device): Device to run the measurement on
        num_iterations (int): Number of iterations to average over
        
    Returns:
        float: Average inference time in milliseconds
    """
    model.eval()
    
    if isinstance(input_size[0], tuple):  # Late fusion
        # For late fusion models, we need to create two dummy inputs
        input1 = torch.randn(1, input_size[0][0], input_size[1], input_size[2]).to(device)
        input2 = torch.randn(1, input_size[0][1], input_size[1], input_size[2]).to(device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(input1, input2)
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input1, input2)
    else:  # Early fusion
        input = torch.randn(1, input_size[0], input_size[1], input_size[2]).to(device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(input)
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input)
    
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000 / num_iterations  # Convert to ms
    
    return inference_time

def calculate_metrics(model, dataloader, criterion, device):
    """
    Calculate comprehensive evaluation metrics for a model.
    
    Args:
        model (nn.Module): PyTorch model
        dataloader (DataLoader): DataLoader for the evaluation data
        criterion (nn.Module): Loss function
        device (torch.device): Device to evaluate on
        
    Returns:
        dict: Dictionary containing all metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Measure inference time
    batch = next(iter(dataloader))
    if len(batch) == 2:  # Early fusion
        inputs, _ = batch
        input_size = (inputs.shape[1], inputs.shape[2], inputs.shape[3])
    else:  # Late fusion
        color_inputs, depth_inputs, _ = batch
        input_size = ((color_inputs.shape[1], depth_inputs.shape[1]), 
                     color_inputs.shape[2], color_inputs.shape[3])
    
    inference_time = measure_inference_time(model, input_size, device)
    
    # Calculate FLOPs
    try:
        flops = calculate_flops(model, input_size, device)
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        flops = 0
    
    # Calculate model parameters and size
    params = count_parameters(model)
    model_size = get_model_size(model)
    
    # Evaluate model performance
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:  # Early fusion
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
            else:  # Late fusion
                color_inputs, depth_inputs, labels = batch
                color_inputs, depth_inputs, labels = color_inputs.to(device), depth_inputs.to(device), labels.to(device)
                outputs = model(color_inputs, depth_inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    
    # For multi-class classification, we need to specify the average method
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate loss
    loss = running_loss / len(dataloader.dataset)
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'parameters': params,
        'model_size_mb': model_size,
        'inference_time_ms': inference_time,
        'flops': flops
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names, output_dir, title='Confusion Matrix'):
    """
    Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
        output_dir (str): Directory to save the plot
        title (str): Title of the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_metrics_history(metrics_history, output_dir):
    """
    Plot metrics history.
    
    Args:
        metrics_history (dict): Dictionary containing metrics history
        output_dir (str): Directory to save the plots
    """
    for metric_name, values in metrics_history.items():
        if metric_name not in ['confusion_matrix', 'parameters', 'model_size_mb', 'inference_time_ms', 'flops']:
            plt.figure(figsize=(10, 6))
            plt.plot(values)
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} vs. Epoch')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{metric_name}_history.png'))
            plt.close()

class MetricsLogger:
    """
    Class for logging metrics to TensorBoard.
    """
    def __init__(self, log_dir):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir (str): Directory to save TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir)
        self.metrics_history = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
    
    def log_metrics(self, metrics, epoch):
        """
        Log metrics to TensorBoard.
        
        Args:
            metrics (dict): Dictionary containing metrics
            epoch (int): Current epoch
        """
        # Log scalar metrics
        for metric_name, value in metrics.items():
            if metric_name not in ['confusion_matrix']:
                self.writer.add_scalar(metric_name, value, epoch)
                
                # Store in history if it's a tracking metric
                if metric_name in self.metrics_history:
                    self.metrics_history[metric_name].append(value)
        
        # Log confusion matrix as an image
        if 'confusion_matrix' in metrics:
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            self.writer.add_figure('confusion_matrix', fig, epoch)
            plt.close(fig)
    
    def close(self):
        """
        Close the TensorBoard writer.
        """
        self.writer.close()