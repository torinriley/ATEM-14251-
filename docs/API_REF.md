# API Reference

### 1. AdaptiveModel


__init__(model_path: str)
- Initialize the adaptive model with a TFLite model path.

predict_next_task(...)
- Predict the next task based on the current task and sensor data.

### 2. ModelTrainer


__init__(tasks_file: str, output_model_path: str)
- Initialize the trainer with tasks and an output model path.


train_and_save_model(epochs: int, batch_size: int)
- Train the model and save it as a TFLite file.

set_max_length(max_length: int)
- Set the maximum sequence length for task encoding and padding.



