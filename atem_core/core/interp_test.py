import numpy as np
import tensorflow as tf
import json

TASK_JSON_PATH = "tasks.json"
TFLITE_MODEL_PATH = "../models/auto_task_optimizer.tflite"

def load_tasks(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data["tasks"]

def create_task_encoder(tasks):
    task_names = sorted(set(task["name"].strip() for task in tasks))
    task_to_index = {name: idx for idx, name in enumerate(task_names)}
    index_to_task = {idx: name for name, idx in task_to_index.items()}
    return task_to_index, index_to_task

def encode_task_order(task_order, task_to_index, max_length):
    encoded_order = [task_to_index[task["name"]] for task in task_order]
    padded_order = np.pad(encoded_order, (0, max_length - len(encoded_order)), constant_values=0)
    return padded_order

def interpret_model(task_order, task_to_index, index_to_task, max_length):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Debug: Show input details
    print("Input Details:", input_details)

    # Encode the task order
    padded_order = encode_task_order(task_order, task_to_index, max_length)

    # Prepare inputs
    input_data_orders = np.array([padded_order], dtype=np.float32)

    # Set inputs and run the model
    interpreter.set_tensor(input_details[0]['index'], input_data_orders)
    interpreter.invoke()

    # Get the predicted score
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_score = output_data[0][0]

    # Map indices back to task names for human-readable task order
    human_readable_tasks = [index_to_task[idx] for idx in padded_order if idx in index_to_task]

    return predicted_score, human_readable_tasks


if __name__ == "__main__":
    # Load tasks and prepare encoder
    tasks = load_tasks(TASK_JSON_PATH)
    task_to_index, index_to_task = create_task_encoder(tasks)

    # Example task order for testing
    max_length = 5
    sample_task_order = [
        {"name": "High Basket", "time": 10},
        {"name": "Low Chamber", "time": 8},
        {"name": "Observation Zone", "time": 5},
    ]

    # Interpret the model
    predicted_score, human_readable_tasks = interpret_model(sample_task_order, task_to_index, index_to_task, max_length)

    print(f"Predicted Score: {predicted_score}")
    print(f"Task Order: {human_readable_tasks}")