import tensorflow as tf
import numpy as np
import json


class AdaptiveModel:
    def __init__(self, model_path):
        """
        Initializes the AdaptiveModel with a TensorFlow Lite interpreter.

        Args:
            model_path (str): Path to the TFLite model file.
        """
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def _prepare_input(self, current_task, sensor_data, task_to_index, max_length):
        """
        Prepares the input tensors for model inference.

        Args:
            current_task (str): Name of the current task.
            sensor_data (dict): Dictionary containing sensor readings.
            task_to_index (dict): Mapping of task names to indices.
            max_length (int): Maximum length for padding.

        Returns:
            tuple: Prepared task and sensor feature arrays.
        """
        encoded_task = [task_to_index[current_task]]
        padded_task = np.pad(encoded_task, (0, max_length - len(encoded_task)), constant_values=0).astype(np.float32).reshape(1, -1)

        sensor_features = np.array([
            sensor_data["time_elapsed"],
            sensor_data["distance_to_target"],
            sensor_data["gyro_angle"],
            sensor_data["battery_level"]
        ], dtype=np.float32).reshape(1, -1)

        return padded_task, sensor_features

    def predict_next_task(self, current_task, sensor_data, task_to_index, index_to_task, max_length):
        """
        Predicts the next task based on the current task and sensor data.

        Args:
            current_task (str): Name of the current task.
            sensor_data (dict): Dictionary containing sensor readings.
            task_to_index (dict): Mapping of task names to indices.
            index_to_task (dict): Mapping of task indices to names.
            max_length (int): Maximum length for padding.

        Returns:
            tuple: The predicted task name and raw output probabilities.
        """
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        padded_task, sensor_features = self._prepare_input(current_task, sensor_data, task_to_index, max_length)

        self.interpreter.set_tensor(input_details[0]['index'], padded_task)
        if len(input_details) > 1:
            self.interpreter.set_tensor(input_details[1]['index'], sensor_features)

        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_index = np.argmax(output)
        return index_to_task[predicted_index], output

    def batch_predict(self, task_sensor_pairs, task_to_index, index_to_task, max_length):
        """
        Predicts next tasks for a batch of inputs.

        Args:
            task_sensor_pairs (list of tuples): List of (current_task, sensor_data) pairs.
            task_to_index (dict): Mapping of task names to indices.
            index_to_task (dict): Mapping of task indices to names.
            max_length (int): Maximum length for padding.

        Returns:
            list of tuples: List of predicted tasks and their raw probabilities.
        """
        predictions = []
        for current_task, sensor_data in task_sensor_pairs:
            predicted_task, raw_output = self.predict_next_task(
                current_task, sensor_data, task_to_index, index_to_task, max_length
            )
            predictions.append((predicted_task, raw_output))
        return predictions

    def save_predictions(self, predictions, output_file="predictions.json"):
        """
        Saves predictions to a JSON file.

        Args:
            predictions (list of tuples): List of predictions to save.
            output_file (str): Path to the JSON file to save.
        """
        formatted_predictions = [
            {"predicted_task": pred[0], "raw_output": pred[1].tolist()} for pred in predictions
        ]
        with open(output_file, "w") as f:
            json.dump(formatted_predictions, f, indent=4)
        print(f"Predictions saved to {output_file}")

    def load_model(self, new_model_path):
        """
        Reloads the model with a new TFLite file.

        Args:
            new_model_path (str): Path to the new TFLite model file.
        """
        self.model_path = new_model_path
        self.interpreter = tf.lite.Interpreter(model_path=new_model_path)
        self.interpreter.allocate_tensors()
        print(f"Model reloaded from {new_model_path}")