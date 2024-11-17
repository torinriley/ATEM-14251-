import json
import os
import numpy as np
import tensorflow as tf
from random import sample

# Paths for task JSON and trained model
TASK_JSON_PATH = "tasks.json"
TFLITE_MODEL_PATH = os.path.join("atem_core", "models", "auto_task_optimizer.tflite")


# Load tasks from JSON file
def load_tasks(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data["tasks"]


# Generate random task orders
def generate_task_orders(tasks, num_samples=500, time_limit=30):
    task_sequences = []
    points = []
    times = []

    for _ in range(num_samples):
        sequence = []
        total_time = 0
        total_points = 0

        for task in sample(tasks, len(tasks)):
            if total_time + task["time"] <= time_limit:
                sequence.append(task)
                total_time += task["time"]
                total_points += task["points"]

        task_sequences.append(sequence)
        points.append(total_points)
        times.append(total_time)

    return task_sequences, points, times


# Create task encoders
def create_task_encoder(tasks):
    task_names = sorted(set(task["name"] for task in tasks))
    task_to_index = {name: idx for idx, name in enumerate(task_names)}
    index_to_task = {idx: name for name, idx in task_to_index.items()}
    return task_to_index, index_to_task


# Encode and pad task sequences
def encode_and_pad_sequences(task_sequences, task_to_index, max_length):
    encoded_sequences = []
    for sequence in task_sequences:
        encoded = [task_to_index[task["name"]] for task in sequence]

        # Truncate sequences that exceed max_length
        if len(encoded) > max_length:
            encoded = encoded[:max_length]

        # Pad sequences to max_length
        padded = np.pad(encoded, (0, max_length - len(encoded)), constant_values=0)
        encoded_sequences.append(padded)

    return np.array(encoded_sequences)


# Train a TensorFlow model
def train_model(X, y, max_length, num_tasks):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=num_tasks, output_dim=16, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X, y, epochs=10, batch_size=16, verbose=1)
    return model


# Predict the optimal sequence
def predict_optimal_sequence(tasks, task_to_index, model, max_length, time_limit=30):
    task_sequences, _, _ = generate_task_orders(tasks, num_samples=1000, time_limit=time_limit)
    encoded_sequences = encode_and_pad_sequences(task_sequences, task_to_index, max_length)

    predictions = model.predict(encoded_sequences)
    best_sequence_idx = np.argmax(predictions)
    best_sequence = task_sequences[best_sequence_idx]
    best_score = predictions[best_sequence_idx]

    return best_sequence, best_score


# Save the trained model as TensorFlow Lite
def save_tflite_model(model, output_path):
    try:
        print("Converting the trained model to TensorFlow Lite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        print(f"Model successfully saved at: {output_path}")
        return True
    except Exception as e:
        print(f"Error occurred during model conversion: {e}")
        return False


# Main Execution
if __name__ == "__main__":
    print("Loading tasks...")
    tasks = load_tasks(TASK_JSON_PATH)

    print("Creating task encoders...")
    task_to_index, index_to_task = create_task_encoder(tasks)

    print("Generating training data...")
    max_length = 5  # Maximum number of tasks in a sequence
    task_sequences, points, times = generate_task_orders(tasks, num_samples=500, time_limit=30)
    X = encode_and_pad_sequences(task_sequences, task_to_index, max_length)
    y = np.array(points)

    print("Training the model...")
    num_tasks = len(task_to_index)
    model = train_model(X, y, max_length, num_tasks)

    print("Predicting the optimal sequence...")
    best_sequence, best_score = predict_optimal_sequence(tasks, task_to_index, model, max_length)
    print("\nOptimal Task Sequence:")
    for task in best_sequence:
        print(f"- {task['name']} (Time: {task['time']}, Points: {task['points']})")
    print(f"Predicted Score: {best_score}")

    # Save the model
    if save_tflite_model(model, TFLITE_MODEL_PATH):
        print("TFLite model saved successfully!")
    else:
        print("Failed to save the TFLite model.")