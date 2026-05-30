import tensorflow as tf
from tensorflow.keras import layers

def get_data_loaders(dataset_dir, img_size=(224, 224), batch_size=32):
    """
    Memuat dataset dari direktori dan membaginya menjadi train, val, dan test sets.
    """

    # Load all images first to split manually for 70/15/15
    base_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    class_names = base_ds.class_names

    # Determine split sizes
    ds_size = len(base_ds)
    train_size = int(0.7 * ds_size)
    val_size = int(0.15 * ds_size)
    test_size = ds_size - train_size - val_size

    train_ds = base_ds.take(train_size)
    remaining = base_ds.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)

    # Data Augmentation Layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    # Apply Augmentation only to train set
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Preprocessing for EfficientNetV2 (Rescaling is usually handled within the model or via specific layer)
    # EfficientNetV2 expects input in range [0, 255] if using the built-in preprocessing layer,
    # but let's be explicit if needed. EfficientNetV2B0 internally handles it if using the Keras application correctly.

    # Prefetching for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
