import os
import tensorflow as tf
import numpy as np
from data_loader import get_data_loaders
from evaluate import evaluate_model
from gradcam import generate_gradcam_for_samples
from utils import create_dirs

def main():
    # Configuration
    DATASET_DIR = 'dataset/'
    MODEL_PATH = 'outputs/model/best_model.keras'
    OUTPUT_CM_DIR = 'outputs/confusion_matrix/'
    OUTPUT_GRADCAM_DIR = 'outputs/gradcam/'

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model tidak ditemukan di {MODEL_PATH}. Silakan jalankan training terlebih dahulu.")
        return

    create_dirs([OUTPUT_CM_DIR, OUTPUT_GRADCAM_DIR])

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # 1. Load Data (Hanya butuh test_ds dan class_names)
    print("Loading data...")
    _, _, test_ds, class_names = get_data_loaders(DATASET_DIR, IMG_SIZE, BATCH_SIZE)

    # 2. Load Existing Model
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 3. Evaluate Model
    print("Evaluating model...")
    evaluate_model(model, test_ds, class_names, OUTPUT_CM_DIR)

    # 4. Grad-CAM Visualization
    # Ambil base_model (layer pertama dari Sequential)
    base_model = model.layers[0]

    # Temukan layer konvolusi terakhir di dalam base_model
    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name:
        print(f"Generating Grad-CAM using layer: {last_conv_layer_name}")
        generate_gradcam_for_samples(model, test_ds, class_names, OUTPUT_GRADCAM_DIR, last_conv_layer_name)
        print(f"Selesai! Hasil Grad-CAM dapat dilihat di: {OUTPUT_GRADCAM_DIR}")
    else:
        print("Error: Tidak dapat menemukan layer konvolusi terakhir.")

if __name__ == "__main__":
    main()
