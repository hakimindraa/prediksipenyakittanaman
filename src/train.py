import os
import tensorflow as tf
from data_loader import get_data_loaders
from model_builder import build_efficientnet_v2, build_mobilenet_v3
from utils import save_training_history, create_dirs
from evaluate import evaluate_model
from gradcam import generate_gradcam_for_samples

def main(model_name="efficientnet"):
    # Configuration
    DATASET_DIR = 'dataset/'
    OUTPUT_MODEL_DIR = 'outputs/model/'
    OUTPUT_PLOT_DIR = f'outputs/plots/{model_name}/'
    OUTPUT_CM_DIR = 'outputs/results/'
    OUTPUT_GRADCAM_DIR = f'outputs/gradcam/{model_name}/'

    create_dirs([OUTPUT_MODEL_DIR, OUTPUT_PLOT_DIR, OUTPUT_CM_DIR, OUTPUT_GRADCAM_DIR])

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20

    # 1. Load Data
    print(f"Loading data for {model_name}...")
    train_ds, val_ds, test_ds, class_names = get_data_loaders(DATASET_DIR, IMG_SIZE, BATCH_SIZE)

    # 2. Build Model
    print(f"Building {model_name} model...")
    if model_name == "efficientnet":
        model, base_model = build_efficientnet_v2(len(class_names))
        save_name = "best_efficientnet.keras"
    elif model_name == "mobilenet":
        model, base_model = build_mobilenet_v3(len(class_names))
        save_name = "best_mobilenet.keras"
    else:
        raise ValueError("model_name must be 'efficientnet' or 'mobilenet'")

    # 3. Training Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUTPUT_MODEL_DIR, save_name),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    # 4. Train Model
    print(f"Starting training {model_name}...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 5. Save History Plots
    save_training_history(history, OUTPUT_PLOT_DIR)

    # 6. Evaluate Model
    print(f"Evaluating {model_name}...")
    evaluate_model(model, test_ds, class_names, OUTPUT_CM_DIR, prefix=model_name)

    # 7. Grad-CAM Visualization
    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name:
        print(f"Generating Grad-CAM for {model_name} using layer: {last_conv_layer_name}")
        generate_gradcam_for_samples(model, test_ds, class_names, OUTPUT_GRADCAM_DIR, last_conv_layer_name)

if __name__ == "__main__":
    # Ubah baris ini sesuai model yang ingin dilatih
    # main(model_name="efficientnet")
    main(model_name="mobilenet")
