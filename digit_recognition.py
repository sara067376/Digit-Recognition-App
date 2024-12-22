import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and normalize the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Create the model
model = tf.keras.models.Sequential()

# Input layer: Flattening the 28x28 image into a 1D array of 784 pixels
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# Adding hidden layers with ReLU activation
model.add(tf.keras.layers.Dense(256, activation='relu'))  # Increased neurons for better feature learning
model.add(tf.keras.layers.Dropout(0.2))  # Dropout to reduce overfitting
model.add(tf.keras.layers.BatchNormalization())  # Batch normalization to stabilize learning

model.add(tf.keras.layers.Dense(128, activation='relu'))  # Second hidden layer
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(64, activation='relu'))  # Third hidden layer
model.add(tf.keras.layers.Dropout(0.1))  # Lower dropout rate to balance learning

# Output layer: 10 neurons for digits 0-9 with softmax for probabilities
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with increased epochs
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save the trained model
model.save('handwritten_improved.keras')
