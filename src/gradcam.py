import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Dapatkan base_model (EfficientNetV2B0)
    base_model = model.layers[0]

    # 2. Cari layer konvolusi terakhir dan semua layer setelahnya di base_model
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Kita buat sub-model untuk bagian awal (sampai konvolusi terakhir)
    conv_model = tf.keras.models.Model(base_model.inputs, last_conv_layer.output)

    # 3. Cari sisa layer di base_model setelah konvolusi terakhir
    # Dan tambahkan layer dari model utama (classifier)

    # Temukan index layer konvolusi di base_model
    conv_idx = -1
    for i, layer in enumerate(base_model.layers):
        if layer.name == last_conv_layer_name:
            conv_idx = i
            break

    # Layer sisa di base_model (seperti top_activation, global_pooling)
    remaining_base_layers = base_model.layers[conv_idx + 1:]
    # Layer di model utama (Sequential) setelah base_model
    classifier_layers = model.layers[1:]

    all_subsequent_layers = remaining_base_layers + classifier_layers

    # 4. Gunakan GradientTape
    with tf.GradientTape() as tape:
        # Dapatkan output konvolusi
        last_conv_output = conv_model(img_array)
        tape.watch(last_conv_output)

        # Jalankan sisa layer satu per satu
        preds = last_conv_output
        for layer in all_subsequent_layers:
            preds = layer(preds)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 5. Hitung gradien
    grads = tape.gradient(class_channel, last_conv_layer_output if 'last_conv_layer_output' in locals() else last_conv_output)

    if grads is None:
        raise ValueError("Gradien bernilai None. Pastikan layer konvolusi terhubung dengan output.")

    # 6. Hitung Heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

def generate_gradcam_for_samples(model, test_ds, class_names, output_dir, last_conv_layer_name):
    for images, labels in test_ds.take(1):
        for i in range(min(5, len(images))):
            img_array = tf.expand_dims(images[i], axis=0)

            preds = model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            true_idx = np.argmax(labels[i])

            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

                temp_path = "temp.jpg"
                tf.keras.preprocessing.image.save_img(temp_path, images[i])

                cam_path = os.path.join(output_dir, f"gradcam_{i}_true_{class_names[true_idx]}_pred_{class_names[pred_idx]}.png")
                save_and_display_gradcam(temp_path, heatmap, cam_path)

                if os.path.exists(temp_path):
                    os.remove(temp_path)
                print(f"Berhasil: {cam_path}")
            except Exception as e:
                print(f"Gagal generate Grad-CAM untuk gambar ke-{i}: {e}")
