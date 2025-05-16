import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re  # For parsing the text files


# --- Helper function to parse all_metrics_summary.txt ---
def parse_late_fusion_summary(file_path):
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()

            # Color Model Metrics
            color_acc_match = re.search(r"Color Model Metrics:.*?accuracy: ([\d.]+)", content,
                                        re.DOTALL | re.IGNORECASE)
            color_params_match = re.search(r"Color Model Metrics:.*?parameters: (\d+)", content,
                                           re.DOTALL | re.IGNORECASE)

            # Depth Model Metrics
            depth_acc_match = re.search(r"Depth Model Metrics:.*?accuracy: ([\d.]+)", content,
                                        re.DOTALL | re.IGNORECASE)
            depth_params_match = re.search(r"Depth Model Metrics:.*?parameters: (\d+)", content,
                                           re.DOTALL | re.IGNORECASE)

            # Fusion Model Metrics
            fusion_acc_match = re.search(r"Fusion Model Metrics:.*?accuracy: ([\d.]+)", content,
                                         re.DOTALL | re.IGNORECASE)

            if color_acc_match: metrics['color_accuracy'] = float(color_acc_match.group(1))
            if color_params_match: metrics['color_parameters'] = int(color_params_match.group(1))
            if depth_acc_match: metrics['depth_accuracy'] = float(depth_acc_match.group(1))
            if depth_params_match: metrics['depth_parameters'] = int(depth_params_match.group(1))
            if fusion_acc_match: metrics['fusion_accuracy'] = float(fusion_acc_match.group(1))

            if 'color_parameters' in metrics and 'depth_parameters' in metrics:
                metrics['total_parameters'] = metrics['color_parameters'] + metrics['depth_parameters']
            else:  # Handle cases where one stream might be missing if parsing fails for it
                metrics['total_parameters'] = metrics.get('color_parameters', 0) + metrics.get('depth_parameters', 0)

            # Calculate efficiencies using Accuracy / Parameters
            if 'fusion_accuracy' in metrics and metrics.get('total_parameters', 0) > 0:
                metrics['fusion_efficiency'] = metrics['fusion_accuracy'] / metrics['total_parameters']
            else:
                metrics['fusion_efficiency'] = 0

            if 'color_accuracy' in metrics and metrics.get('color_parameters', 0) > 0:
                metrics['color_efficiency'] = metrics['color_accuracy'] / metrics['color_parameters']
            else:
                metrics['color_efficiency'] = 0

            if 'depth_accuracy' in metrics and metrics.get('depth_parameters', 0) > 0:
                metrics['depth_efficiency'] = metrics['depth_accuracy'] / metrics['depth_parameters']
            else:
                metrics['depth_efficiency'] = 0

            metrics['file_path'] = file_path

    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
    return metrics


# --- Define your directories (ensure these paths are correct relative to script location) ---
base_dir = "."

unweighted_dirs_paths = [
    "results/late_fusion_20250511_152953", "results/late_fusion_20250511_165953",
    "results/late_fusion_20250512_143931", "results/late_fusion_20250512_150309",
    "results/late_fusion_20250512_164107", "results/late_fusion_20250512_175256",
    "results/late_fusion_20250512_175852", "results/late_fusion_20250512_183756",
    "results/late_fusion_20250514_112025", "results/late_fusion_20250514_113312"
]

weighted_dirs_paths = [
    "results/late_fusion_20250512_185641_start_of_weighted_averaging", "results/late_fusion_20250512_190658",
    "results/late_fusion_20250512_194222", "results/late_fusion_20250512_195308",
    "results/late_fusion_20250512_200108", "results/late_fusion_20250513_181720",
    "results/late_fusion_20250513_182755", "results/late_fusion_20250513_183244",
    "results/late_fusion_20250513_183737", "results/late_fusion_20250513_202144",
    "results/late_fusion_20250513_202723", "results/late_fusion_20250513_203333",
    "results/late_fusion_20250513_204003", "results/late_fusion_20250513_204715",
    "results/late_fusion_20250513_205310", "results/late_fusion_20250513_210122",
    "results/late_fusion_20250513_210843", "results/late_fusion_20250513_211646",
    "results/late_fusion_20250513_212457", "results/late_fusion_20250513_213242"
]


# --- Process Late Fusion Data ---
def load_late_fusion_data(dir_paths, fusion_type_label):
    data = []
    for dir_path in dir_paths:
        summary_file = os.path.join(base_dir, dir_path, "all_metrics_summary.txt")
        if os.path.exists(summary_file):
            metrics = parse_late_fusion_summary(summary_file)
            if metrics:  # Ensure metrics were successfully parsed
                metrics['type'] = fusion_type_label
                data.append(metrics)
        else:
            print(f"Warning: Summary file not found at {summary_file}")
    return pd.DataFrame(data)


df_unweighted_late = load_late_fusion_data(unweighted_dirs_paths, 'Unweighted Late Fusion')
df_weighted_late = load_late_fusion_data(weighted_dirs_paths, 'Weighted Late Fusion')

df_late_fusion_all = pd.concat([df_unweighted_late, df_weighted_late], ignore_index=True)

metrics = ['fusion_accuracy', 'total_parameters']

# Compute descriptive statistics by fusion type
grouped = df_late_fusion_all.groupby('type')[metrics].describe()

unstacked = grouped.unstack(level=0)
if isinstance(unstacked, pd.Series):
    summary = unstacked.swaplevel(0, 1).sort_index(level=0)
else:
    summary = unstacked.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
print("\nLate Fusion Metrics Summary:")
print(summary)

early_fusion_runs_data = [
    {'Run': 1, 'Accuracy': 0.9792, 'Parameters': 171560, 'Time': 192.32, 'type': 'Early Fusion'},
    {'Run': 2, 'Accuracy': 0.9792, 'Parameters': 164839, 'Time': 234.36, 'type': 'Early Fusion'},
    {'Run': 3, 'Accuracy': 0.9729, 'Parameters': 821672, 'Time': 173.19, 'type': 'Early Fusion'},
    {'Run': 4, 'Accuracy': 0.9729, 'Parameters': 2579865, 'Time': 269.82, 'type': 'Early Fusion'},
    {'Run': 5, 'Accuracy': 0.9750, 'Parameters': 213683, 'Time': 230.01, 'type': 'Early Fusion'},
    {'Run': 6, 'Accuracy': 0.9771, 'Parameters': 900677, 'Time': 165.38, 'type': 'Early Fusion'},
    {'Run': 7, 'Accuracy': 0.9646, 'Parameters': 3776387, 'Time': 324.08, 'type': 'Early Fusion'},
    {'Run': 8, 'Accuracy': 0.9646, 'Parameters': 4132127, 'Time': 275.76, 'type': 'Early Fusion'},
    {'Run': 9, 'Accuracy': 0.9646, 'Parameters': 1543239, 'Time': 299.92, 'type': 'Early Fusion'},
    {'Run': 10, 'Accuracy': 0.9688, 'Parameters': 758468, 'Time': 210.71, 'type': 'Early Fusion'},
    {'Run': 11, 'Accuracy': 0.9688, 'Parameters': 3397358, 'Time': 261.17, 'type': 'Early Fusion'},
    {'Run': 12, 'Accuracy': 0.9604, 'Parameters': 2444747, 'Time': 202.62, 'type': 'Early Fusion'},
    {'Run': 13, 'Accuracy': 0.9750, 'Parameters': 759296, 'Time': 235.58, 'type': 'Early Fusion'},
    {'Run': 14, 'Accuracy': 0.9563, 'Parameters': 1508084, 'Time': 180.62, 'type': 'Early Fusion'},
    {'Run': 15, 'Accuracy': 0.9708, 'Parameters': 2314184, 'Time': 288.57, 'type': 'Early Fusion'},
    {'Run': 16, 'Accuracy': 0.9708, 'Parameters': 2295635, 'Time': 247.03, 'type': 'Early Fusion'},
    {'Run': 17, 'Accuracy': 0.9688, 'Parameters': 3516040, 'Time': 340.57, 'type': 'Early Fusion'},
    {'Run': 18, 'Accuracy': 0.9750, 'Parameters': 520760, 'Time': 159.37, 'type': 'Early Fusion'},
    {'Run': 19, 'Accuracy': 0.9646, 'Parameters': 2324739, 'Time': 257.81, 'type': 'Early Fusion'},
    {'Run': 20, 'Accuracy': 0.9646, 'Parameters': 774269, 'Time': 186.92, 'type': 'Early Fusion'}
]
df_early_fusion = pd.DataFrame(early_fusion_runs_data)
if not df_early_fusion.empty:
    df_early_fusion['Efficiency'] = df_early_fusion.apply(
        lambda row: row['Accuracy'] / row['Parameters'] if row['Parameters'] > 0 else 0, axis=1
    )
    df_early_fusion['type'] = 'Early Fusion'


# --- Function to create summary table for a given DataFrame ---
def create_summary_table(df, accuracy_col, params_col, efficiency_col, model_name_label):
    if df is None or df.empty or not all(col in df.columns for col in [accuracy_col, params_col, efficiency_col]):
        print(f"Could not create summary table for {model_name_label}: DataFrame is empty or missing required columns.")
        # Return a DataFrame with N/A values if data is missing
        return pd.DataFrame({
            'Run Type': ['Highest Accuracy', 'Most Efficient', 'Lowest Parameters', 'Average (all runs)'],
            'Accuracy': ['N/A'] * 4,
            'Parameters': ['N/A'] * 4,
            'Efficiency ($Acc/P$)': ['N/A'] * 4
        })

    # Ensure numeric types for calculation
    df[accuracy_col] = pd.to_numeric(df[accuracy_col], errors='coerce')
    df[params_col] = pd.to_numeric(df[params_col], errors='coerce')
    df[efficiency_col] = pd.to_numeric(df[efficiency_col], errors='coerce')

    df_cleaned = df.dropna(subset=[accuracy_col, params_col, efficiency_col])
    if df_cleaned.empty:
        print(f"Could not create summary table for {model_name_label} after cleaning NaNs.")
        return pd.DataFrame({
            'Run Type': ['Highest Accuracy', 'Most Efficient', 'Lowest Parameters', 'Average (all runs)'],
            'Accuracy': ['N/A'] * 4,
            'Parameters': ['N/A'] * 4,
            'Efficiency ($Acc/P$)': ['N/A'] * 4
        })

    highest_acc_run = df_cleaned.loc[df_cleaned[accuracy_col].idxmax()]
    most_efficient_run = df_cleaned.loc[df_cleaned[efficiency_col].idxmax()]
    lowest_params_run = df_cleaned.loc[df_cleaned[params_col].idxmin()]
    average_metrics = df_cleaned[[accuracy_col, params_col, efficiency_col]].mean()

    summary_data = {
        'Run Type': ['Highest Accuracy', 'Most Efficient', 'Lowest Parameters', 'Average (all runs)'],
        'Accuracy': [highest_acc_run[accuracy_col], most_efficient_run[accuracy_col], lowest_params_run[accuracy_col],
                     average_metrics[accuracy_col]],
        'Parameters': [int(highest_acc_run[params_col]), int(most_efficient_run[params_col]),
                       int(lowest_params_run[params_col]), int(average_metrics[params_col])],
        'Efficiency ($Acc/P$)': [highest_acc_run[efficiency_col], most_efficient_run[efficiency_col],
                                 lowest_params_run[efficiency_col], average_metrics[efficiency_col]]
    }
    return pd.DataFrame(summary_data)


# --- Generate and Print Summary Tables ---
print("\n--- psoCNN Early Fusion Summary ---")
early_fusion_summary_table = create_summary_table(df_early_fusion, 'Accuracy', 'Parameters', 'Efficiency',
                                                  'Early Fusion')
print(early_fusion_summary_table.to_string(index=False, formatters={'Accuracy': '{:.4f}'.format,
                                                                    'Efficiency ($Acc/P$)': '{:.3e}'.format}))

# Extract average metrics for early fusion
pso_early_avg_row = early_fusion_summary_table[early_fusion_summary_table['Run Type'] == 'Average (all runs)']
if not pso_early_avg_row.empty:
    pso_early_avg_acc = pso_early_avg_row['Accuracy'].values[0]
    pso_early_avg_params = pso_early_avg_row['Parameters'].values[0]
    pso_early_avg_efficiency = pso_early_avg_row['Efficiency ($Acc/P$)'].values[0]
else:  # Fallback
    pso_early_avg_acc = 0.9650  # Example fallback value
    pso_early_avg_params = 1500000  # Example fallback value
    pso_early_avg_efficiency = pso_early_avg_acc / pso_early_avg_params if pso_early_avg_params > 0 else 0

# For Late Fusion, we use df_weighted_late for Color, Depth, and Total Weighted Fusion summaries
print("\n--- psoCNN Late Fusion Color Stream Summary (from Weighted Experiments) ---")
color_stream_summary_table = create_summary_table(df_weighted_late, 'color_accuracy', 'color_parameters',
                                                  'color_efficiency', 'Late Fusion Color Stream')
print(color_stream_summary_table.to_string(index=False, formatters={'Accuracy': '{:.4f}'.format,
                                                                    'Efficiency ($Acc/P$)': '{:.3e}'.format}))

print("\n--- psoCNN Late Fusion Depth Stream Summary (from Weighted Experiments) ---")
depth_stream_summary_table = create_summary_table(df_weighted_late, 'depth_accuracy', 'depth_parameters',
                                                  'depth_efficiency', 'Late Fusion Depth Stream')
print(depth_stream_summary_table.to_string(index=False, formatters={'Accuracy': '{:.4f}'.format,
                                                                    'Efficiency ($Acc/P$)': '{:.3e}'.format}))

print("\n--- psoCNN Weighted Late Fusion System Summary ---")
weighted_late_fusion_summary_table = create_summary_table(df_weighted_late, 'fusion_accuracy', 'total_parameters',
                                                          'fusion_efficiency', 'Weighted Late Fusion System')
print(weighted_late_fusion_summary_table.to_string(index=False, formatters={'Accuracy': '{:.4f}'.format,
                                                                            'Efficiency ($Acc/P$)': '{:.3e}'.format}))

# Extract average metrics for weighted late fusion
pso_weighted_late_avg_row = weighted_late_fusion_summary_table[weighted_late_fusion_summary_table['Run Type'] == 'Average (all runs)']
if not pso_weighted_late_avg_row.empty:
    pso_weighted_late_avg_acc = pso_weighted_late_avg_row['Accuracy'].values[0]
    pso_weighted_late_avg_params = pso_weighted_late_avg_row['Parameters'].values[0]
    pso_weighted_late_avg_efficiency = pso_weighted_late_avg_row['Efficiency ($Acc/P$)'].values[0]
else:  # Fallback
    pso_weighted_late_avg_acc = 0.9200  # Example fallback value
    pso_weighted_late_avg_params = 700000  # Example fallback value
    pso_weighted_late_avg_efficiency = pso_weighted_late_avg_acc / pso_weighted_late_avg_params if pso_weighted_late_avg_params > 0 else 0

# --- Main Comparison Table (Baseline vs. Most Efficient psoCNN) ---
# Baseline parameters (update if you have a more precise T_tool_classes)
T_tool_classes = 4  # Assuming 4 action classes for baseline's action-specific head
baseline_early_params = 2193880  # From user's table
baseline_late_params = 4239536  # From user's table
baseline_early_acc = 0.9900
baseline_late_acc = 0.9900
baseline_early_efficiency = baseline_early_acc / baseline_early_params if baseline_early_params > 0 else 0
baseline_late_efficiency = baseline_late_acc / baseline_late_params if baseline_late_params > 0 else 0

# Get most efficient psoCNN models
# User provided these values:
# psoCNN Early Fusion (Most Efficient) 0.979200 164839
# psoCNN Unweighted Late Fusion (Most Efficient) 0.735417 739146
# psoCNN Weighted Late Fusion (Most Efficient) 0.945833 659592

pso_early_efficient_acc = 0.979200
pso_early_efficient_params = 164839
pso_early_efficient_efficiency = pso_early_efficient_acc / pso_early_efficient_params if pso_early_efficient_params > 0 else 0

# For unweighted late, find the most efficient from df_unweighted_late
most_efficient_unweighted_run = None
if not df_unweighted_late.empty and 'fusion_efficiency' in df_unweighted_late.columns and not df_unweighted_late[
    'fusion_efficiency'].isnull().all():
    most_efficient_unweighted_run = df_unweighted_late.loc[df_unweighted_late['fusion_efficiency'].idxmax()]
    pso_unweighted_late_efficient_acc = most_efficient_unweighted_run['fusion_accuracy']
    pso_unweighted_late_efficient_params = int(most_efficient_unweighted_run['total_parameters'])
    pso_unweighted_late_efficient_efficiency = most_efficient_unweighted_run['fusion_efficiency']
else:  # Fallback to user provided if parsing/data is incomplete
    pso_unweighted_late_efficient_acc = 0.735417
    pso_unweighted_late_efficient_params = 739146
    pso_unweighted_late_efficient_efficiency = pso_unweighted_late_efficient_acc / pso_unweighted_late_efficient_params if pso_unweighted_late_efficient_params > 0 else 0

# For weighted late, find the most efficient from df_weighted_late
most_efficient_weighted_run = None
if not df_weighted_late.empty and 'fusion_efficiency' in df_weighted_late.columns and not df_weighted_late[
    'fusion_efficiency'].isnull().all():
    most_efficient_weighted_run = df_weighted_late.loc[df_weighted_late['fusion_efficiency'].idxmax()]
    pso_weighted_late_efficient_acc = most_efficient_weighted_run['fusion_accuracy']
    pso_weighted_late_efficient_params = int(most_efficient_weighted_run['total_parameters'])
    pso_weighted_late_efficient_efficiency = most_efficient_weighted_run['fusion_efficiency']
else:  # Fallback
    pso_weighted_late_efficient_acc = 0.945833
    pso_weighted_late_efficient_params = 659592
    pso_weighted_late_efficient_efficiency = pso_weighted_late_efficient_acc / pso_weighted_late_efficient_params if pso_weighted_late_efficient_params > 0 else 0

main_comparison_data = {
    'Model Type': [
        'Baseline Early Fusion (Friezas et al. inferred)',
        'Baseline Late Fusion (Friezas et al. inferred)',
        'psoCNN Early Fusion (Most Efficient)',
        'psoCNN Early Fusion (Average of All Runs)',
        'psoCNN Weighted Late Fusion (Most Efficient)',
        'psoCNN Weighted Late Fusion (Average of All Runs)'

    ],
    'Accuracy': [
        baseline_early_acc,
        baseline_late_acc,
        pso_early_efficient_acc,
        pso_early_avg_acc,
        pso_weighted_late_efficient_acc,
        pso_weighted_late_avg_acc if 'pso_weighted_late_avg_acc' in locals() else 0.0
    ],
    'Parameters': [
        baseline_early_params,
        baseline_late_params,
        pso_early_efficient_params,
        pso_early_avg_params,
        pso_weighted_late_efficient_params,
        pso_weighted_late_avg_params
    ],
    'Efficiency ($Acc/P$)': [
        baseline_early_efficiency,
        baseline_late_efficiency,
        pso_early_efficient_efficiency,
        pso_early_avg_efficiency,
        pso_weighted_late_efficient_efficiency,
        pso_weighted_late_avg_efficiency
    ]
}
df_main_comparison = pd.DataFrame(main_comparison_data)
print("\n--- Main Comparison Table (Baseline vs. Most Efficient psoCNN) ---")
print(df_main_comparison.to_string(index=False, formatters={'Accuracy': '{:.4f}'.format, 'Parameters': '{:,}'.format,
                                                            'Efficiency ($Acc/P$)': '{:.3e}'.format}))
df_main_comparison.to_csv("main_comparison_table.csv", index=False)

# --- Boxplots (Optional - can be uncommented if needed) ---
plt.style.use('seaborn-v0_8-whitegrid')

# Prepare data for boxplots
late_fusion_boxplot_data = None
if not df_late_fusion_all.empty and 'fusion_accuracy' in df_late_fusion_all.columns and not df_late_fusion_all[
    'fusion_accuracy'].isnull().all():
    # Create a copy for the boxplot data
    late_fusion_boxplot_data = df_late_fusion_all[['type', 'fusion_accuracy', 'total_parameters']].copy()

    # Boxplot: Weighted vs. Unweighted Late Fusion Accuracy
    plt.figure(figsize=(7, 6))
    sns.boxplot(x='type', y='fusion_accuracy', data=df_late_fusion_all, palette=['white', 'white'],
                showmeans=True, meanline=True, meanprops={'color': 'red', 'ls': '-', 'lw': 2},
                medianprops={'color': 'black'}, boxprops={'edgecolor': 'black'},
                whiskerprops={'color': 'black'}, capprops={'color': 'black'})
    plt.xlabel('Late Fusion Type', fontsize=12)
    plt.ylabel('Fusion Accuracy', fontsize=12)
    plt.xticks(fontsize=10, rotation=0)
    plt.yticks(fontsize=10)
    min_val = df_late_fusion_all['fusion_accuracy'].min()
    max_val = df_late_fusion_all['fusion_accuracy'].max()
    plt.ylim(max(0, min_val - 0.05), min(1, max_val + 0.05))
    plt.tight_layout()
    plt.savefig("/Users/mihnea/_workspace_/_uni/thesis/docs/boxplots/boxplot_weighted_vs_unweighted_late_fusion.png",
                dpi=300)
    plt.show()
else:
    print("\nSkipping boxplot for weighted vs unweighted late fusion: Not enough data or all NaNs.")

# Prepare early vs weighted fusion data for boxplots
early_weighted_fusion_boxplot_data = None
if not df_early_fusion.empty and not df_weighted_late.empty and \
        'Accuracy' in df_early_fusion.columns and 'fusion_accuracy' in df_weighted_late.columns and \
        not df_early_fusion['Accuracy'].isnull().all() and not df_weighted_late['fusion_accuracy'].isnull().all():

    df_early_for_plot = df_early_fusion[['Accuracy', 'Parameters', 'type']].copy()
    df_weighted_late_for_plot = df_weighted_late[['fusion_accuracy', 'total_parameters', 'type']].copy()
    df_weighted_late_for_plot.rename(columns={'fusion_accuracy': 'Accuracy', 'total_parameters': 'Parameters'},
                                     inplace=True)

    early_weighted_fusion_boxplot_data = pd.concat([df_early_for_plot, df_weighted_late_for_plot], ignore_index=True)

    # Boxplot: Early Fusion vs. Weighted Late Fusion Accuracy
    plt.figure(figsize=(7, 6))
    sns.boxplot(x='type', y='Accuracy', data=early_weighted_fusion_boxplot_data, palette=['white', 'white'],
                showmeans=True, meanline=True, meanprops={'color': 'red', 'ls': '-', 'lw': 2},
                medianprops={'color': 'black'}, boxprops={'edgecolor': 'black'},
                whiskerprops={'color': 'black'}, capprops={'color': 'black'})
    plt.xlabel('psoCNN Fusion Strategy', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(fontsize=10, rotation=0)
    plt.yticks(fontsize=10)
    min_val = early_weighted_fusion_boxplot_data['Accuracy'].min()
    max_val = early_weighted_fusion_boxplot_data['Accuracy'].max()
    plt.ylim(max(0, min_val - 0.05), min(1, max_val + 0.05))
    plt.tight_layout()
    plt.savefig("/Users/mihnea/_workspace_/_uni/thesis/docs/boxplots/boxplot_early_vs_weighted_late_fusion.png",
                dpi=300)
    plt.show()
else:
    print("\nSkipping boxplot for early vs weighted late fusion: Not enough data or all NaNs.")

# --- Generate tables for boxplot data to explain results ---
print("\n--- Detailed Data for Boxplots ---")

# Table 1: Late Fusion Accuracy Data (Weighted vs. Unweighted)
if late_fusion_boxplot_data is not None:
    print("\n1. Late Fusion Accuracy Data (Weighted vs. Unweighted):")

    # Calculate statistics per fusion type
    late_fusion_stats = late_fusion_boxplot_data.groupby('type')['fusion_accuracy'].describe().round(4)
    print(late_fusion_stats)

    # Show all individual data points used in the boxplot
    print("\nIndividual Data Points:")
    for fusion_type in late_fusion_boxplot_data['type'].unique():
        filtered_data = late_fusion_boxplot_data[late_fusion_boxplot_data['type'] == fusion_type]
        print(f"\n{fusion_type}:")
        accuracy_values = filtered_data['fusion_accuracy'].sort_values(ascending=False).reset_index(drop=True)
        param_values = filtered_data['total_parameters'].values
        fusion_data_table = pd.DataFrame({
            'Run': range(1, len(accuracy_values) + 1),
            'Accuracy': accuracy_values,
            'Parameters': param_values
        })
        print(fusion_data_table.to_string(index=False, formatters={
            'Accuracy': '{:.4f}'.format,
            'Parameters': '{:,}'.format
        }))

# Table 2: Early vs. Weighted Late Fusion Data
if early_weighted_fusion_boxplot_data is not None:
    print("\n2. Early Fusion vs. Weighted Late Fusion Data:")

    # Calculate statistics per fusion type
    early_weighted_stats = early_weighted_fusion_boxplot_data.groupby('type')['Accuracy'].describe().round(4)
    print(early_weighted_stats)

    # Show all individual data points used in the boxplot
    print("\nIndividual Data Points:")
    for fusion_type in early_weighted_fusion_boxplot_data['type'].unique():
        filtered_data = early_weighted_fusion_boxplot_data[early_weighted_fusion_boxplot_data['type'] == fusion_type]
        print(f"\n{fusion_type}:")
        accuracy_values = filtered_data['Accuracy'].sort_values(ascending=False).reset_index(drop=True)
        param_values = filtered_data['Parameters'].values
        fusion_data_table = pd.DataFrame({
            'Run': range(1, len(accuracy_values) + 1),
            'Accuracy': accuracy_values,
            'Parameters': param_values
        })
        print(fusion_data_table.to_string(index=False, formatters={
            'Accuracy': '{:.4f}'.format,
            'Parameters': '{:,}'.format
        }))
