import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import create_dirs

def compare():
    RESULTS_DIR = 'outputs/results/'
    COMPARISON_DIR = 'outputs/comparison/'
    create_dirs([COMPARISON_DIR])

    # 1. Load Performance Metrics (Accuracy, etc)
    models = ['efficientnet', 'mobilenet']
    perf_data = []

    for model in models:
        json_path = os.path.join(RESULTS_DIR, f'{model}_metrics.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metrics = json.load(f)

            perf_data.append({
                'Model': model,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['macro avg']['precision'],
                'Recall': metrics['macro avg']['recall'],
                'F1-score': metrics['macro avg']['f1-score']
            })

    if not perf_data:
        print("Error: Metrics performance tidak ditemukan.")
        return

    df_perf = pd.DataFrame(perf_data)

    # 2. Load Analysis Metrics (Size, Inference)
    analysis_path = os.path.join(COMPARISON_DIR, 'model_analysis.csv')
    if os.path.exists(analysis_path):
        df_analysis = pd.read_csv(analysis_path)
        # Merge data
        df_complete = pd.merge(df_perf, df_analysis[['Model', 'Size_MB', 'Avg_Inference_ms']], on='Model')
    else:
        print("Warning: model_analysis.csv belum ada. Jalankan src/model_analysis.py dulu.")
        df_complete = df_perf

    # Save Complete CSV
    df_complete.to_csv(os.path.join(COMPARISON_DIR, 'model_comparison_complete.csv'), index=False)

    # Plotting
    sns.set_style("whitegrid")

    # 1. Performance Comparison Plot
    df_melted = df_perf.melt(id_vars='Model', var_name='Metric', value_name='Value')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Metric', y='Value', hue='Model')
    plt.title('Performance Comparison')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'performance_comparison.png'))
    plt.close()

    # 2. Size Comparison Plot
    if 'Size_MB' in df_complete.columns:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df_complete, x='Model', y='Size_MB')
        plt.title('Model Size Comparison (MB)')
        plt.savefig(os.path.join(COMPARISON_DIR, 'model_size_comparison.png'))
        plt.close()

    # 3. Inference Comparison Plot
    if 'Avg_Inference_ms' in df_complete.columns:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df_complete, x='Model', y='Avg_Inference_ms')
        plt.title('Average Inference Time (ms)')
        plt.savefig(os.path.join(COMPARISON_DIR, 'inference_time_comparison.png'))
        plt.close()

    # Summary Generation
    best_acc_row = df_complete.loc[df_complete['Accuracy'].idxmax()]
    best_f1_row = df_complete.loc[df_complete['F1-score'].idxmax()]

    summary_text = f"""==================================
MODEL COMPARISON SUMMARY
==================================

Model terbaik berdasarkan Accuracy:
{best_acc_row['Model']} ({best_acc_row['Accuracy']:.4f})

Model terbaik berdasarkan F1-score:
{best_f1_row['Model']} ({best_f1_row['F1-score']:.4f})
"""

    if 'Size_MB' in df_complete.columns:
        smallest_row = df_complete.loc[df_complete['Size_MB'].idxmin()]
        summary_text += f"\nModel paling ringan:\n{smallest_row['Model']} ({smallest_row['Size_MB']:.2f} MB)\n"

    if 'Avg_Inference_ms' in df_complete.columns:
        fastest_row = df_complete.loc[df_complete['Avg_Inference_ms'].idxmin()]
        summary_text += f"\nModel tercepat:\n{fastest_row['Model']} ({fastest_row['Avg_Inference_ms']:.2f} ms)\n"

    summary_text += "=================================="

    print("\n" + summary_text)

    with open(os.path.join(COMPARISON_DIR, 'comparison_summary.txt'), 'w') as f:
        f.write(summary_text)

if __name__ == "__main__":
    compare()
