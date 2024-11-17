import tensorflow as tf
import numpy as np

# Example training data
X_train = np.array([
    [0, 10, 1.5, 45, 90],  # Example: [current_task, time_elapsed, distance, gyro_angle, battery]
    [1, 20, 0.8, 30, 80],
    [2, 15, 0.5, 10, 70],
])
y_train = np.array([1, 2, 0])  # Next task indices

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")  # Output probabilities for each task
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=20)

# Save to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("adaptive_model.tflite", "wb") as f:
    f.write(tflite_model)





