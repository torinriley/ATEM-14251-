import tensorflow as tf
import numpy as np

class AdaptiveModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def predict_next_task(self, current_task, sensor_data, task_to_index, index_to_task, max_length):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        encoded_task = [task_to_index[current_task]]
        padded_task = np.pad(encoded_task, (0, max_length - len(encoded_task)), constant_values=0).astype(np.float32).reshape(1, -1)

        sensor_features = np.array([
            sensor_data["time_elapsed"],
            sensor_data["distance_to_target"],
            sensor_data["gyro_angle"],
            sensor_data["battery_level"]
        ], dtype=np.float32).reshape(1, -1)

        self.interpreter.set_tensor(input_details[0]['index'], padded_task)
        if len(input_details) > 1:
            self.interpreter.set_tensor(input_details[1]['index'], sensor_features)

        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_index = np.argmax(output)
        return index_to_task[predicted_index], output