import json
import numpy as np
import tensorflow as tf
from train2 import create_task_encoder

# Paths for the task JSON and TFLite model
TASK_JSON_PATH = "tasks.json"
TFLITE_MODEL_PATH = "../models/auto_task_optimizer.tflite"


def interpret_model(current_task, sensor_data, task_to_index, index_to_task, max_length):
    """
    Interpret the model to predict the next task based on current task and sensor data.

    Args:
        current_task (str): The name of the current task.
        sensor_data (dict): Sensor data input for the model.
        task_to_index (dict): Mapping of task names to indices.
        index_to_task (dict): Mapping of indices to task names.
        max_length (int): Maximum length for task encoding.

    Returns:
        tuple: Predicted task and the associated task probabilities.
    """
    print("Loading TensorFlow Lite model...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Encode the current task
    encoded_task = [task_to_index[current_task]]
    padded_task = np.pad(encoded_task, (0, max_length - len(encoded_task)), constant_values=0).astype(np.float32).reshape(1, -1)

    # Format sensor data
    sensor_features = np.array([
        sensor_data["time_elapsed"],
        sensor_data["distance_to_target"],
        sensor_data["gyro_angle"],
        sensor_data["battery_level"]
    ], dtype=np.float32).reshape(1, -1)

    # Set input tensors
    interpreter.set_tensor(input_details[0]['index'], padded_task)
    if len(input_details) > 1:
        interpreter.set_tensor(input_details[1]['index'], sensor_features)

    # Run inference
    interpreter.invoke()

    # Get model output
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(output)
    predicted_task = index_to_task[predicted_index]

    return predicted_task, output


def simulate_real_world():
    """
    Simulate real-world scenarios for testing the adaptive model.
    """
    print("Loading tasks...")
    tasks = json.load(open(TASK_JSON_PATH))["tasks"]

    print("Creating task encoders...")
    task_to_index, index_to_task = create_task_encoder(tasks)

    max_length = 5
    current_task = list(task_to_index.keys())[0]  # Start with the first task
    task_sequence = []

    print("Starting competition simulation...\n")
    for iteration in range(5):  # Simulate 5 task iterations
        print(f"Iteration {iteration + 1}:")

        # Simulate sensor data
        sensor_data = {
            "time_elapsed": int(np.random.uniform(10, 30)),
            "distance_to_target": round(np.random.uniform(0.5, 2.0), 2),
            "gyro_angle": int(np.random.uniform(0, 180)),
            "battery_level": int(np.random.uniform(50, 100))
        }

        print(f"Current Task: {current_task}")
        print(f"Sensor Data: {sensor_data}")

        # Call interpret_model
        predicted_task, predicted_scores = interpret_model(
            current_task,
            sensor_data,
            task_to_index,
            index_to_task,
            max_length
        )

        print(f"Next Task Predicted: {predicted_task}")
        print(f"Predicted Task Scores: {predicted_scores}")

        task_sequence.append(current_task)
        current_task = predicted_task

        print("-" * 50)

    print("\nFinal Task Sequence:")
    print(task_sequence)

    # Write results to JSON
    output = {
        "final_task_sequence": task_sequence
    }
    with open("real_time_task_sequence.json", "w") as json_file:
        json.dump(output, json_file, indent=4)

    print("\nResults written to 'real_time_task_sequence.json'")
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    simulate_real_world()