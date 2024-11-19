import os
import pytest
from atem.model_train import ModelTrainer

# Sample tasks.json content for testing
TASKS_JSON = {
    "tasks": [
        {"name": "Task 1", "time": 5, "points": 10},
        {"name": "Task 2", "time": 10, "points": 20}
    ]
}

@pytest.fixture
def sample_tasks_file(tmp_path):
    """Fixture to create a temporary tasks.json file."""
    tasks_file = tmp_path / "tasks.json"
    with open(tasks_file, "w") as f:
        import json
        json.dump(TASKS_JSON, f)
    return str(tasks_file)

def test_model_trainer_initialization(sample_tasks_file):
    trainer = ModelTrainer(tasks_file=sample_tasks_file)
    assert trainer.tasks is not None
    assert len(trainer.tasks) == 2

def test_generate_training_data(sample_tasks_file):
    trainer = ModelTrainer(tasks_file=sample_tasks_file)
    task_sequences, points, sensor_data = trainer._generate_training_data()
    assert len(task_sequences) > 0
    assert len(points) > 0
    assert len(sensor_data) > 0

def test_train_and_save_model(sample_tasks_file, tmp_path):
    output_model_path = tmp_path / "model.tflite"
    trainer = ModelTrainer(tasks_file=sample_tasks_file, output_model_path=str(output_model_path))
    trainer.train_and_save_model(epochs=1, batch_size=2)
    assert os.path.exists(output_model_path)
