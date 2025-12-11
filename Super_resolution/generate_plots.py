import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import re
import numpy as np
import seaborn as sns

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    # Fallback if specific style not found
    plt.style.use('seaborn-darkgrid')

def load_experiment_data(data_dir):
    """
    Loads experiment data for comparison plots.
    Replicates logic from notebook: reads epoch_metrics.csv, 
    calculates mean val_psnr (as 'PSNR') and mean duration (as 'Runtime_min').
    """
    results = []
    # Search for epoch_metrics.csv recursively
    training_files = glob.glob(os.path.join(data_dir, '**', 'epoch_metrics.csv'), recursive=True)
    
    for file_path in training_files:
        parent_dir = os.path.basename(os.path.dirname(file_path))
        # Match 'Scale_0.20' or similar
        match = re.search(r'Scale_(\d+\.\d+)', parent_dir, re.IGNORECASE)
        
        if match:
            scale = float(match.group(1))
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                
                if 'val_psnr' in df.columns and 'duration_s' in df.columns:
                    # Note: Notebook comment said "Use max val_psnr" but code used .mean()
                    # We replicate the code behavior.
                    max_psnr = df['val_psnr'].mean()
                    # Use average duration per epoch as runtime metric
                    avg_epoch_min = df['duration_s'].mean() / 60.0
                    
                    results.append({
                        'Scale': scale,
                        'PSNR': max_psnr,
                        'Runtime_min': avg_epoch_min
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values('Scale')

def load_eval_metrics(data_dir):
    """
    Loads evaluation metrics for PSNR vs Scale plots.
    Reads eval_metrics.csv and calculates mean psnr_y.
    """
    results = []
    files = glob.glob(os.path.join(data_dir, '**', 'eval_metrics.csv'), recursive=True)
    
    for file_path in files:
        match = re.search(r'Scale_(\d+\.\d+)', file_path)
        if match:
            scale = float(match.group(1))
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                if 'psnr_y' in df.columns:
                    avg_psnr = df['psnr_y'].mean()
                    results.append({'Scale': scale, 'Avg_PSNR': avg_psnr})
            except Exception:
                pass
    
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values('Scale')

def plot_training_curves(data_dir, exp_name, metric_col, ylabel, title, filename):
    """
    Plots a metric (e.g., loss, val_psnr) vs Epoch for all scales in an experiment.
    """
    files = glob.glob(os.path.join(data_dir, '**', 'epoch_metrics.csv'), recursive=True)
    
    if not files:
        print(f"No epoch_metrics.csv files found in {data_dir}")
        return

    plt.figure(figsize=(10, 6))
    
    # Sort files by scale for consistent legend order
    files_with_scale = []
    for fp in files:
        match = re.search(r'Scale_(\d+\.\d+)', fp)
        if match:
            files_with_scale.append((float(match.group(1)), fp))
    files_with_scale.sort()
    
    plotted = False
    for scale, file_path in files_with_scale:
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            
            if metric_col in df.columns:
                # Assuming 'epoch' column exists, otherwise use index
                x = df['epoch'] if 'epoch' in df.columns else df.index + 1
                plt.plot(x, df[metric_col], label=f'Scale {scale}')
                plotted = True
        except Exception as e:
            print(f"Error plotting {file_path}: {e}")
            
    if plotted:
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{title} ({exp_name})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()
    else:
        print(f"No data found for metric '{metric_col}' in {exp_name}")

def plot_runtime_vs_scale(data_dir, exp_name, filename):
    """
    Plots Total Training Time and Average Epoch Time vs Scale.
    """
    df = load_experiment_data(data_dir) # Reusing this as it has Runtime_min
    
    if df.empty:
        print(f"No data for runtime plot in {exp_name}")
        return

    # We need total duration too, which load_experiment_data calculates as avg.
    # Let's re-calculate or just use the avg for the "Average Epoch Time" plot
    # and maybe skip Total Time if we don't strictly need it or re-implement loading.
    # The notebook plotted BOTH. Let's do a quick re-load to get total time if needed,
    # or just plot Average Epoch Time which is in df['Runtime_min'].
    
    # Let's stick to Average Epoch Time as it's cleaner and already loaded.
    # If Total Time is strictly required, we can add it. 
    # Notebook Step 50 showed "Total Training Time" and "Average Epoch Time".
    # Let's implement a specific loader for this to be precise.
    
    results = []
    files = glob.glob(os.path.join(data_dir, '**', 'epoch_metrics.csv'), recursive=True)
    for file_path in files:
        match = re.search(r'Scale_(\d+\.\d+)', file_path)
        if match:
            scale = float(match.group(1))
            try:
                d = pd.read_csv(file_path)
                d.columns = d.columns.str.strip()
                if 'duration_s' in d.columns:
                    total_min = d['duration_s'].sum() / 60.0
                    avg_min = d['duration_s'].mean() / 60.0
                    results.append({
                        'Scale': scale,
                        'Total_Min': total_min,
                        'Avg_Min': avg_min
                    })
            except: pass
    
    if not results:
        return

    rdf = pd.DataFrame(results).sort_values('Scale')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(rdf['Scale'], rdf['Total_Min'], 'o-', linewidth=2)
    ax1.set_xlabel('Scale')
    ax1.set_ylabel('Total Runtime (minutes)')
    ax1.set_title(f'Total Training Time vs Scale ({exp_name})')
    ax1.grid(True, linestyle='--', alpha=0.2)
    
    ax2.plot(rdf['Scale'], rdf['Avg_Min'], 'o-', linewidth=2)
    ax2.set_xlabel('Scale')
    ax2.set_ylabel('Minutes per epoch')
    ax2.set_title(f'Average Epoch Time vs Scale ({exp_name})')
    ax2.grid(True, linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

def plot_psnr_vs_scale_eval(data_dir, exp_name, filename):
    """
    Plots Average Evaluation PSNR vs Scale using eval_metrics.csv
    """
    df = load_eval_metrics(data_dir)
    if df.empty:
        print(f"No eval metrics found for {exp_name}")
        return
        
    plt.figure(figsize=(8, 6))
    plt.plot(df['Scale'], df['Avg_PSNR'], 'o-', linewidth=2, label=exp_name)
    plt.xlabel('Scale')
    plt.ylabel('Average PSNR (dB)')
    plt.title(f'Average Evaluation PSNR vs Scale ({exp_name})')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

def plot_comparisons(data_dir1, data_dir2):
    """
    Generates comparison plots between Exp 1 and Exp 2.
    """
    print("Loading Experiment 1 data for comparison...")
    df1 = load_experiment_data(data_dir1)
    print("Loading Experiment 2 data for comparison...")
    df2 = load_experiment_data(data_dir2)
    
    if df1.empty or df2.empty:
        print("Insufficient data for comparison.")
        return

    # Merge
    df_comp = pd.merge(df1, df2, on='Scale', suffixes=('_Exp1', '_Exp2'))
    
    if df_comp.empty:
        print("No overlapping scales for comparison.")
        return

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    width = 0.35
    x = np.arange(len(df_comp['Scale']))
    
    # PSNR Comparison
    ax1.bar(x - width/2, df_comp['PSNR_Exp1'], width, label='Exp 1 (Fixed)', color='#895798')
    ax1.bar(x + width/2, df_comp['PSNR_Exp2'], width, label='Exp 2 (Adaptive)', color='#386cb0')
    ax1.set_xlabel('Scale')
    ax1.set_ylabel('Max Validation PSNR (dB)')
    ax1.set_title('PSNR Comparison: Fixed vs Adaptive Depth', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comp['Scale'])
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.2, axis='y')
    
    # Runtime Comparison
    ax2.bar(x - width/2, df_comp['Runtime_min_Exp1'], width, label='Exp 1 (Fixed)', color='#895798')
    ax2.bar(x + width/2, df_comp['Runtime_min_Exp2'], width, label='Exp 2 (Adaptive)', color='#386cb0')
    ax2.set_xlabel('Scale')
    ax2.set_ylabel('Average Time per Epoch (minutes)')
    ax2.set_title('Runtime Comparison: Fixed vs Adaptive Depth', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_comp['Scale'])
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_plots.png')
    print("Saved comparison_plots.png")
    plt.close()

def plot_model_size(csv_path):
    """
    Plots Model Size vs Scale.
    """
    if not os.path.exists(csv_path):
        print(f"Model size CSV not found at {csv_path}")
        return
        
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df = df.sort_values('Scale')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['Scale'], df['Size_MB'], 'o-', color='#d62728', linewidth=2, markersize=8)
        
        plt.xlabel('Scale')
        plt.ylabel('Model Size (MB)')
        plt.title('Model Size vs Scale (Experiment 2)')
        plt.grid(True, linestyle='--', alpha=0.4)
        
        for x, y in zip(df['Scale'], df['Size_MB']):
            plt.annotate(f'{y:.1f} MB', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
        plt.savefig('model_size_vs_scale.png')
        print("Saved model_size_vs_scale.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting model size: {e}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_exp1 = os.path.join(base_dir, 'final_data', 'Experiment_1')
    data_dir_exp2 = os.path.join(base_dir, 'final_data', 'Experiment_2')
    model_size_csv = os.path.join(data_dir_exp2, 'model_size.csv')
    
    print("Generating Experiment 1 Plots...")
    plot_training_curves(data_dir_exp1, 'Experiment 1', 'loss', 'Training Loss', 'Training Loss vs Epoch', 'exp1_loss_vs_epoch.png')
    plot_training_curves(data_dir_exp1, 'Experiment 1', 'val_psnr', 'PSNR (dB)', 'Validation PSNR vs Epoch', 'exp1_psnr_vs_epoch.png')
    plot_runtime_vs_scale(data_dir_exp1, 'Experiment 1', 'exp1_runtime_vs_scale.png')
    plot_psnr_vs_scale_eval(data_dir_exp1, 'Experiment 1', 'exp1_eval_psnr_vs_scale.png')
    
    print("\nGenerating Experiment 2 Plots...")
    plot_training_curves(data_dir_exp2, 'Experiment 2', 'loss', 'Training Loss', 'Training Loss vs Epoch', 'exp2_loss_vs_epoch.png')
    plot_training_curves(data_dir_exp2, 'Experiment 2', 'val_psnr', 'PSNR (dB)', 'Validation PSNR vs Epoch', 'exp2_psnr_vs_epoch.png')
    plot_runtime_vs_scale(data_dir_exp2, 'Experiment 2', 'exp2_runtime_vs_scale.png')
    plot_psnr_vs_scale_eval(data_dir_exp2, 'Experiment 2', 'exp2_eval_psnr_vs_scale.png')
    
    print("\nGenerating Comparison Plots...")
    plot_comparisons(data_dir_exp1, data_dir_exp2)
    
    print("\nGenerating Model Size Plot...")
    plot_model_size(model_size_csv)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
