import json
import numpy as np
import tensorflow as tf
from train import create_task_encoder, generate_task_orders

# Paths for the task JSON and TFLite model
TASK_JSON_PATH = "tasks.json"
TFLITE_MODEL_PATH = "models/auto_task_optimizer.tflite"

def interpret_model(task_order, task_to_index, index_to_task, max_length):
    """
    Interpret the TFLite model to get the predicted score and task names for a given task order.
    """
    print("Loading TensorFlow Lite model...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Encode and pad the task order
    encoded_order = [task_to_index[task["name"]] for task in task_order]
    padded_order = np.pad(encoded_order, (0, max_length - len(encoded_order)), constant_values=0)

    # Prepare input data
    input_data = np.array([padded_order], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()
    predicted_score = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Map indices back to task names
    task_names = [index_to_task[idx] for idx in encoded_order]

    return float(predicted_score), task_names  # Convert NumPy float32 to Python float


if __name__ == "__main__":
    print("Loading tasks...")
    tasks = json.load(open(TASK_JSON_PATH))["tasks"]

    print("Creating task encoders...")
    task_to_index, index_to_task = create_task_encoder(tasks)

    print("Generating a sample task order...")
    max_length = 5  # Example max length
    sample_task_order, _, _ = generate_task_orders(tasks, num_samples=1, time_limit=30)

    # Use the first sample task order
    sample_task_order = sample_task_order[0]

    print("Interpreting the model...")
    predicted_score, task_names = interpret_model(sample_task_order, task_to_index, index_to_task, max_length)

    # Prepare JSON output
    output = {
        "predicted_score": predicted_score,  # Python float is JSON serializable
        "task_order": task_names
    }

    # Save to a JSON file
    output_path = "interpreted_task_order.json"
    with open(output_path, "w") as json_file:
        json.dump(output, json_file, indent=4)

    print(f"Results written to {output_path}")
    print(json.dumps(output, indent=4))