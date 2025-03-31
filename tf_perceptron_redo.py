# Author: Joshua Krasnogorov

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

data = pd.read_csv("histograms.txt", sep=' ', header=None)
valid_data = pd.read_csv("histograms_validation.txt", sep=' ', header=None)

# split features and labels
X = data.iloc[:, 2:-1].to_numpy() # I appended 1 to my original histogram files to account for bias, removing it for tf
y = data.iloc[:, 0].to_numpy()

X_valid = valid_data.iloc[:, 2:-1].to_numpy()
y_valid = valid_data.iloc[:, 0].to_numpy()

# split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

batch_size = 100
epochs = 100
learning_rate = 0.1

# convert labels to categorical (for binary classification)
y_train = np.where(y_train == -1, 0, 1)
y_test = np.where(y_test == -1, 0, 1)
y_valid = np.where(y_valid == -1, 0, 1)

# define the model creation function
def create_model(activation='sigmoid'):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(768,)),  # 768 input neurons
        tf.keras.layers.Dense(1000, activation=activation), # first hidden layer
        tf.keras.layers.Dense(1000, activation=activation), # second hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid') # output layer
    ])
    return model

# TensorBoard callback
tensorboard_callback_sigmoid = tf.keras.callbacks.TensorBoard(log_dir='./logs/sigmoid', histogram_freq=1)
tensorboard_callback_relu = tf.keras.callbacks.TensorBoard(log_dir='./logs/relu', histogram_freq=1)

# function to train and evaluate model
def train_and_evaluate(activation, callback):
    model = create_model(activation)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # train the model
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        callbacks=[callback],
                        verbose=1)
    
    # evaluate on validation set
    valid_loss, valid_accuracy = model.evaluate(X_valid, y_valid, verbose=0)
    print(f"\n{activation.upper()} Activation:")
    print(f"Validation Loss: {valid_loss:.4f}")
    print(f"Validation Accuracy: {valid_accuracy:.4f}")
    
    return model, history

# train with Sigmoid activation
print("Training with Sigmoid Activation...")
model_sigmoid, history_sigmoid = train_and_evaluate('sigmoid', tensorboard_callback_sigmoid)

# train with ReLU activation
print("\nTraining with ReLU Activation...")
model_relu, history_relu = train_and_evaluate('relu', tensorboard_callback_relu)