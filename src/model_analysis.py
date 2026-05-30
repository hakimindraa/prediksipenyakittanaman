import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from utils import create_dirs

def get_model_size(model_path):
    """Menghitung ukuran file model dalam MB."""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)

def benchmark_inference(model, img_array, n_iter=100):
    """Mengukur waktu inferensi rata-rata, min, dan max."""
    # Warmup
    for _ in range(10):
        _ = model.predict(img_array, verbose=0)

    times = []
    for _ in range(n_iter):
        start_time = time.time()
        _ = model.predict(img_array, verbose=0)
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # ke ms

    return np.mean(times), np.min(times), np.max(times)

def main():
    MODEL_DIR = 'outputs/model/'
    DATASET_DIR = 'dataset/'
    COMPARISON_DIR = 'outputs/comparison/'
    create_dirs([COMPARISON_DIR])

    models_info = {
        'efficientnet': os.path.join(MODEL_DIR, 'best_efficientnet.keras'),
        'mobilenet': os.path.join(MODEL_DIR, 'best_mobilenet.keras')
    }

    # Ambil 1 gambar contoh dari dataset
    sample_img_path = None
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                sample_img_path = os.path.join(root, file)
                break
        if sample_img_path: break

    if not sample_img_path:
        print("Error: Tidak ditemukan gambar di folder dataset.")
        return

    # Preprocess image
    img = Image.open(sample_img_path).resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    results = []

    for name, path in models_info.items():
        if not os.path.exists(path):
            print(f"Warning: Model {name} tidak ditemukan di {path}")
            continue

        print(f"Menganalisis {name}...")

        # 1. Size
        size_mb = get_model_size(path)

        # 2. Inference
        model = tf.keras.models.load_model(path)
        avg_ms, min_ms, max_ms = benchmark_inference(model, img_array)

        results.append({
            'Model': name,
            'Size_MB': round(size_mb, 2),
            'Avg_Inference_ms': round(avg_ms, 2),
            'Min_Inference_ms': round(min_ms, 2),
            'Max_Inference_ms': round(max_ms, 2)
        })

        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Avg Inference: {avg_ms:.2f} ms")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(COMPARISON_DIR, 'model_analysis.csv'), index=False)

    # Graphics
    sns.set_style("whitegrid")

    # Plot Size
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Model', y='Size_MB', palette='Set2')
    plt.title('Model Size Comparison (MB)')
    plt.savefig(os.path.join(COMPARISON_DIR, 'model_size_comparison.png'))
    plt.close()

    # Plot Inference
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Model', y='Avg_Inference_ms', palette='Set1')
    plt.title('Average Inference Time (ms)')
    plt.savefig(os.path.join(COMPARISON_DIR, 'inference_time_comparison.png'))
    plt.close()

    print("\nAnalisis Model Selesai. Hasil disimpan di outputs/comparison/")

if __name__ == "__main__":
    main()
