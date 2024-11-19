import json
import tensorflow as tf
import numpy as np

class Interpreter:
    def __init__(self, model_path):
        """
        Initialize the interpreter with the path to a TFLite model.

        Args:
            model_path (str): Path to the TensorFlow Lite model file.
        """
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def interpret(self, task_sequence, sensor_data, max_length=5):
        """
        Interpret the model based on the provided inputs.

        Args:
            task_sequence (list): Sequence of tasks encoded as integers.
            sensor_data (list): List of sensor data values.
            max_length (int): Maximum length of the task sequence.

        Returns:
            dict: Predicted task index and output scores.
        """
        # Prepare input tensors
        encoded_task = np.pad(task_sequence, (0, max_length - len(task_sequence)), constant_values=0).astype(np.float32).reshape(1, -1)
        sensor_features = np.array(sensor_data, dtype=np.float32).reshape(1, -1)

        self.interpreter.set_tensor(self.input_details[0]['index'], encoded_task)
        if len(self.input_details) > 1:
            self.interpreter.set_tensor(self.input_details[1]['index'], sensor_features)

        # Run the model
        self.interpreter.invoke()

        # Get the output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        predicted_task_index = int(np.argmax(output))

        return {"predicted_task_index": predicted_task_index, "output_scores": output.tolist()}

    def save_interpretation(self, interpretation, output_path="interpretation.json"):
        """
        Save the interpretation result to a JSON file.

        Args:
            interpretation (dict): Dictionary containing the interpretation results.
            output_path (str): Path to save the JSON file.
        """
        with open(output_path, "w") as json_file:
            json.dump(interpretation, json_file, indent=4)
        print(f"Interpretation saved to {output_path}")
