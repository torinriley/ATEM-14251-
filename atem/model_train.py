import tensorflow as tf
import numpy as np
import json
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class ModelTrainer:
    def __init__(self, tasks_file, output_model_path="trained_model.tflite"):
        """
        Initialize the ModelTrainer with a tasks file and output model path.

        Args:
            tasks_file (str): Path to the JSON file containing tasks.
            output_model_path (str): Path to save the trained TFLite model.
        """
        self.tasks_file = tasks_file
        self.output_model_path = output_model_path
        self.tasks = self._load_tasks()
        self.task_to_index, self.index_to_task = self._create_task_encoder()
        self.max_length = 5  # Default maximum sequence length

    def _load_tasks(self):
        """
        Load tasks from the JSON file.

        Returns:
            list: A list of tasks.
        """
        print(f"{Fore.BLUE}Loading tasks from {self.tasks_file}...")
        with open(self.tasks_file, "r") as file:
            data = json.load(file)
        print(f"{Fore.GREEN}Tasks loaded successfully!")
        return data["tasks"]

    def _create_task_encoder(self):
        """
        Create task encoders for mapping task names to indices and vice versa.

        Returns:
            tuple: task_to_index and index_to_task mappings.
        """
        print(f"{Fore.BLUE}Creating task encoder...")
        task_names = sorted(set(task["name"] for task in self.tasks))
        task_to_index = {name: idx for idx, name in enumerate(task_names)}
        index_to_task = {idx: name for name, idx in task_to_index.items()}
        print(f"{Fore.GREEN}Task encoder created successfully!")
        return task_to_index, index_to_task

    def _generate_training_data(self, num_samples=500, time_limit=30):
        """
        Generate training data from tasks.

        Args:
            num_samples (int): Number of task sequences to generate.
            time_limit (int): Maximum time limit for a sequence.

        Returns:
            tuple: Encoded task sequences, corresponding points, and sensor data.
        """
        task_sequences = []
        points = []
        sensor_data = []

        for _ in range(num_samples):
            sequence = []
            total_time = 0
            total_points = 0
            while total_time < time_limit:
                task = np.random.choice(self.tasks)
                if total_time + task["time"] <= time_limit:
                    sequence.append(task["name"])
                    total_time += task["time"]
                    total_points += task["points"]

                    # Generate mock sensor data
                    sensor_data.append({
                        "time_elapsed": np.random.randint(10, 30),
                        "distance_to_target": np.random.uniform(0.5, 2.0),
                        "gyro_angle": np.random.randint(0, 180),
                        "battery_level": np.random.randint(50, 100)
                    })

            task_sequences.append(sequence)
            points.append(total_points)

        return task_sequences, points, sensor_data

    def _encode_and_pad_sequences(self, task_sequences):
        """
        Encode and pad task sequences for model training.

        Args:
            task_sequences (list): List of task sequences.

        Returns:
            np.array: Encoded and padded task sequences.
        """
        encoded_sequences = []
        for sequence in task_sequences:
            # Ensure all tasks are valid
            for task in sequence:
                if task not in self.task_to_index:
                    raise ValueError(f"Task '{task}' not found in task_to_index mapping.")

            # Encode the sequence
            encoded = [self.task_to_index[task] for task in sequence]

            # Truncate if sequence length exceeds max_length
            if len(encoded) > self.max_length:
                encoded = encoded[:self.max_length]

            # Pad to max_length
            padded = np.pad(encoded, (0, self.max_length - len(encoded)), constant_values=0)
            encoded_sequences.append(padded)

        return np.array(encoded_sequences)

    def train_and_save_model(self, epochs=20, batch_size=16):
        """
        Train a model and save it in TFLite format.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        print("Generating training data...")
        task_sequences, points, sensor_data = self._generate_training_data()

        print("Encoding and padding sequences...")
        X = self._encode_and_pad_sequences(task_sequences)
        y = np.array(points)

        print("Building the model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=len(self.task_to_index), output_dim=16, input_length=self.max_length),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear")
        ])

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        print(f"{Fore.BLUE}Training the model for {epochs} epochs...")
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

        print(f"{Fore.BLUE}Converting the model to TensorFlow Lite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        print(f"{Fore.GREEN}Saving the model to {self.output_model_path}...")
        with open(self.output_model_path, "wb") as file:
            file.write(tflite_model)

        print(f"{Fore.GREEN}Model training and saving complete!")

    def set_max_length(self, new_max_length):
        """
        Update the maximum sequence length for padding.

        Args:
            new_max_length (int): New maximum sequence length.
        """
        print(f"{Fore.YELLOW}Updating max_length from {self.max_length} to {new_max_length}...")
        self.max_length = new_max_length
        print(f"{Fore.GREEN}max_length updated successfully!")

    def get_task_mappings(self):
        """
        Get task-to-index and index-to-task mappings.

        Returns:
            tuple: task_to_index and index_to_task mappings.
        """
        return self.task_to_index, self.index_to_task