import tensorflow as tf
from tensorflow.keras import layers, models

def build_efficientnet_v2(num_classes, input_shape=(224, 224, 3)):
    """
    Membangun model EfficientNetV2B0 dengan Transfer Learning.
    """
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

def build_mobilenet_v3(num_classes, input_shape=(224, 224, 3)):
    """
    Membangun model MobileNetV3Small dengan Transfer Learning.
    """
    base_model = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model
