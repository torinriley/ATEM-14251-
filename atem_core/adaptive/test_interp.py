import json
import numpy as np
import tensorflow as tf
from time import sleep
from random import uniform, choice

from atem_core.core.train import create_task_encoder

TASK_JSON_PATH = "tasks.json"
TFLITE_MODEL_PATH = "../models/auto_task_optimizer.tflite"

TASK_COMPLETION_STATUSES = ["completed", "failed"]

def interpret_model(current_task, sensor_data, task_to_index, index_to_task, max_length):
    """
    Interpret the model to predict the next task based on current task and sensor data.
    """
    print("Loading TensorFlow Lite model...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    encoded_task = [task_to_index[current_task]]
    padded_task = np.pad(encoded_task, (0, max_length - len(encoded_task)), constant_values=0).astype(np.float32).reshape(1, -1)

    sensor_features = np.array([
        sensor_data["time_elapsed"],
        sensor_data["distance_to_target"],
        sensor_data["gyro_angle"],
        sensor_data["battery_level"]
    ], dtype=np.float32).reshape(1, -1)

    interpreter.set_tensor(input_details[0]['index'], padded_task)

    if len(input_details) > 1:
        interpreter.set_tensor(input_details[1]['index'], sensor_features)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(output)
    predicted_task = index_to_task[predicted_index]

    return predicted_task, output

def simulate_real_world():
    """
    Simulate real-world competition scenarios.
    """
    tasks = json.load(open(TASK_JSON_PATH))["tasks"]
    task_to_index, index_to_task = create_task_encoder(tasks)

    max_length = 5
    task_sequence = []
    current_task = choice(list(task_to_index.keys()))
    iteration = 1

    print("Starting competition simulation...\n")

    while len(task_sequence) < 5:
        print(f"Iteration {iteration}:")

        # Simulate sensor data
        sensor_data = {
            "time_elapsed": int(uniform(10, 30)),
            "distance_to_target": round(uniform(0.5, 2.0), 2),
            "gyro_angle": int(uniform(0, 180)),
            "battery_level": int(uniform(50, 100))
        }

        print(f"Current Task: {current_task}")
        print(f"Sensor Data: {sensor_data}")

        predicted_task, predicted_scores = interpret_model(
            current_task,
            sensor_data,
            task_to_index,
            index_to_task,
            max_length
        )

        print(f"Next Task Predicted: {predicted_task}")
        print(f"Predicted Task Scores: {predicted_scores}")

        task_status = choice(TASK_COMPLETION_STATUSES)
        print(f"Task Completion Status: {task_status}")

        if task_status == "completed":
            task_sequence.append(current_task)
            current_task = predicted_task
        else:
            print(f"Task '{current_task}' failed. Retrying...\n")
            continue

        print("-" * 50)
        iteration += 1
        sleep(1)

    print("\nFinal Task Sequence:")
    print(task_sequence)

    output = {
        "final_task_sequence": task_sequence
    }

    with open("real_time_task_sequence.json", "w") as json_file:
        json.dump(output, json_file, indent=4)

    print("\nResults written to 'real_time_task_sequence.json'")
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    simulate_real_world()