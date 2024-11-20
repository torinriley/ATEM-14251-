# API Reference

### 1. AdaptiveModel


__init__(model_path: str)
- Initialize the adaptive model with a TFLite model path.


predict_next_task(task_sequence: list, sensor_data: list)
- Predict the next task based on the current task and sensor data.


save_interpretation(interpretation: dict, output_path: str = "interpretation.json")
- Save the interpretation result to a JSON file.

### 2. ModelTrainer


__init__(tasks_file: str, output_model_path: str)
- Initialize the trainer with tasks and an output model path.


train_and_save_model(epochs: int, batch_size: int)
- Train the model and save it as a TFLite file.

set_max_length(max_length: int)
- Set the maximum sequence length for task encoding and padding.



### 3. Interpreter

__init__(model_path: str)

•	Initialize the interpreter with the path to a TFLite model.

interpret(task_sequence: list, sensor_data: list, max_length: int = 5)

•	Perform interpretation of the model based on task sequence and sensor data.
**Arguments:**
•	task_sequence: List of tasks encoded as integers.

•	sensor_data: List of sensor data values.

•	max_length: Maximum length of the task sequence (default: 5).

*Returns:**

•	dict: Contains the predicted task index and output scores.

save_interpretation(interpretation: dict, output_path: str = "interpretation.json")

•	Save the interpretation result to a JSON file.

**Arguments:**

•	interpretation: A dictionary containing the interpretation results.

•	output_path: Path to save the JSON file (default: "interpretation.json").
