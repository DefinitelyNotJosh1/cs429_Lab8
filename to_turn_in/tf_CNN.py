# Author: Joshua Krasnogorov

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import glob

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# load images
def load_images_from_folder(folder, label, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images = []
    labels = []
    for filename in glob.glob(f"{folder}/*.png"):
        img = load_img(filename, target_size=img_size)
        img_array = img_to_array(img) / 255.0 # normalize
        images.append(img_array)
        labels.append(label)
    return images, labels

# load "yes" and "no" images
yes_images, yes_labels = load_images_from_folder("./yes", 1)
no_images, no_labels = load_images_from_folder("./no", 0)

# combine datasets
X = np.array(yes_images + no_images)
y = np.array(yes_labels + no_labels)

# split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=101)


def create_cnn_model(conv_layers=3, filters=[32, 64, 128], kernel_size=(3, 3)):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    ])
    
    # add convolutional layers
    for i in range(conv_layers):
        model.add(tf.keras.layers.Conv2D(filters[i], kernel_size, activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    # flatten and dense layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))  # Prevent overfitting
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Binary output
    
    return model

# different configurations
configurations = [
    {"conv_layers": 3, "filters": [32, 64, 128], "kernel_size": (3, 3)}, # default
    {"conv_layers": 3, "filters": [16, 32, 64], "kernel_size": (5, 5)}, # larger kernel size
    {"conv_layers": 2, "filters": [32, 64], "kernel_size": (3, 3)}, # fewer layers
]

# TensorBoard
log_dir = "./logs/cnn_experiments"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# train
for idx, config in enumerate(configurations):
    print(f"\nTraining Configuration {idx + 1}: {config}")
    
    model = create_cnn_model(conv_layers=config["conv_layers"],
                            filters=config["filters"],
                            kernel_size=config["kernel_size"])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_val, y_val),
                        callbacks=[tensorboard_callback],
                        verbose=1)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
