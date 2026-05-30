import tensorflow as tf
import numpy as np

CLASS_NAMES = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_healthy']

def preprocess_image(image, target_size=(224, 224)):
    """
    Menyiapkan gambar untuk prediksi.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    # EfficientNetV2 handles rescaling internally if using the Keras application
    return img_array

def get_prediction(model, img_array):
    """
    Melakukan prediksi dan mengembalikan label serta confidence.
    """
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx]

    all_probs = {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}

    return CLASS_NAMES[pred_idx], confidence, all_probs
