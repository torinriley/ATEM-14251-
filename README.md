
# ATEM - Adaptive Task Execution Model

---

# Getting Started with ATEM
Welcome to ATEM, the adaptive task execution and machine learning package designed for FTC robotics and beyond. Follow this quick-start guide to get up and running with ATEM in your project.

## 1. Install ATEM

```bash 
pip install atem
```

## 2. Train a Model in Python
ATEM provides a Python API to train a TensorFlow Lite model based on your tasks.

- Create a tasks.json file to define the tasks your robot will perform. Here’s an example:
```json
{
  "tasks": [
    { "name": "Observation Zone", "time": 5, "points": 3 },
    { "name": "Net Zone", "time": 4, "points": 2 },
    { "name": "Low Basket", "time": 5, "points": 4 },
    { "name": "High Basket", "time": 7, "points": 9 }
  ]
}
```
- Train a Model
Use the following Python script to train and save a TensorFlow Lite model:

```python
from atem.model_train import ModelTrainer

# Paths to tasks file and output model
tasks_file = "tasks.json"
output_model_path = "adaptive_model.tflite"

# Initialize ModelTrainer
trainer = ModelTrainer(tasks_file=tasks_file, output_model_path=output_model_path)

# Train the model and save it
trainer.train_and_save_model(epochs=20, batch_size=16)

print("Task-to-index mappings:")
print(trainer.get_task_mappings())
```

## 3. Integrate the Model in Java
ATEM models can be integrated into Java-based FTC projects using TensorFlow Lite for inference.

- **Step 1: Add TensorFlow Lite Dependencies**

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.11.0'
}
```

- **Step 2: Include the Trained Model in Your Project**
Place the adaptive_model.tflite file in the assets directory of your FTC project:

`TeamCode/src/main/assets/adaptive_model.tflite`

- **Step 3: Implement Model Inference**
Use the following Java code to interpret the trained model:

```java
import org.tensorflow.lite.Interpreter;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

public class ATEMModelInterpreter {
    private final Interpreter interpreter;

    public ATEMModelInterpreter(String modelPath) throws Exception {
        ByteBuffer modelBuffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(modelPath)));
        this.interpreter = new Interpreter(modelBuffer);
    }

    public String predictNextTask(String currentTask, Map<String, Double> sensorData, 
                                  Map<String, Integer> taskToIndex, Map<Integer, String> indexToTask, int maxLength) {
        // Implement inference logic here using ATEM-trained model
        return "Low Basket"; // Replace with actual implementation
    }
}
```


## 4. Example Usage in an FTC Project
Use the trained model to predict and execute tasks dynamically.

- **Step 1: Initialize the Model Interpreter**
```java
ATEMModelInterpreter modelInterpreter = new ATEMModelInterpreter("adaptive_model.tflite");
```

- **Step 2: Use Predictions to Execute Tasks**
```java
String nextTask = modelInterpreter.predictNextTask(currentTask, sensorData, taskToIndex, indexToTask, 5);
System.out.println("Predicted Next Task: " + nextTask);
```
## All set!

---




## **Features**
- **Model Training**:
  - Train a TensorFlow model to optimize task sequences for maximum points.
  - Tasks and their respective points are dynamically loaded from a JSON file.
  - Outputs a TensorFlow Lite model for lightweight deployment.

- **Model Interpretation**:
  - Given a list of tasks, predicts the optimal sequence and total points.
  - Outputs human-readable task orders and scores.

---

## **How It Works**
### 1. **Task JSON File**
The `tasks.json` file defines the tasks available for the autonomous phase:
```json
{
  "tasks": [
    { "name": "High Basket", "points": 10, "time": 5 },
    { "name": "Low Basket", "points": 5, "time": 3 },
    "..."
  ]
}
```
## **Training the Model**

The model uses task data to train on sequences of tasks for maximizing points within a time limit:
- Loads tasks from the tasks.json file. 
- Generates random task sequences within the given time constraint. 
- Encodes tasks and trains a model to predict scores based on sequences. 
- Outputs a TensorFlow Lite model for deployment.


## **Interpreting the Model**

The interpreter script takes a sequence of tasks, predicts the total points, and outputs the best sequence in human-readable format.


## **Technical Details**

**Model Architecture**

**Input:**
- Task indices (embedded into dense vectors).
- Task times (numeric values).
- Hidden Layers:
- Dense layers for feature extraction and sequence analysis.

- **Output**
- Predicted total points for a given task sequence.

**Data Encoding**

- Task names are encoded as numerical indices.
- Task times are padded to a fixed length for uniform input.

---

# Adaptive Task Prediction Model

---

## Overview
The Adaptive Task Prediction Model is designed to enable real-time decision-making for autonomous robots. It processes sensor data after each task completion, predicts the next optimal task, and adjusts its strategy based on the robot’s current state and environmental feedback.

This dynamic approach ensures the robot maximizes performance, conserves resources, and adapts to unexpected changes in real-world scenarios.

---

## Workflow

### 1. **Sensor Data Collection**
After completing each task, the robot gathers sensor data to provide a snapshot of its current state:
- **Time Elapsed**: Time taken to complete the task.
- **Distance to Target**: The robot's proximity to the next goal.
- **Gyro Angle**: Orientation relative to the reference.
- **Battery Level**: Remaining energy for task prioritization.
- Additional sensor inputs like vision or LIDAR can be incorporated.

---

### 2. **Feature Encoding**
Sensor data and the current task ID are encoded into a format compatible with the machine learning model:
- Continuous values are normalized for consistent input ranges.
- Categorical values are converted to embeddings or indices.

---

### 3. **Real-Time Model Inference**
The model processes the encoded input to:
1. **Predict the Next Task**:
   - Outputs the most likely task to maximize performance.
2. **Provide Task Scores**:
   - Confidence levels for all possible tasks.

**Example**:
```plaintext
Input:
- Current Task: "Observation Zone"
- Sensor Data: {time_elapsed: 20, distance_to_target: 0.5, gyro_angle: 45, battery_level: 70}

Output:
- Predicted Next Task: "High Basket"
- Task Scores: [0.1, 0.8, 0.1]
```

## Model Inferencing
The Adaptive Task Prediction Model utilizes a TensorFlow Lite (TFLite) model for efficient inference. This lightweight, optimized model is specifically designed for resource-constrained environments like robotics systems, ensuring fast and accurate predictions in real time.

---

### **The model requires encoded inputs representing:**
- Current Task: Encoded as a numerical ID using the task_to_index mapping.
- sensor Data: Real-time inputs such as:
- time_elapsed: Normalized elapsed time.
- distance_to_target: Scaled distance to the next target.
- gyro_angle: Angle, normalized to a fixed range.
- battery_level: Percentage value normalized between 0 and 1.

*The inputs are padded to match the model’s expected dimensions if needed.

### **Once the input data is prepared, it is passed into the TFLite interpreter:**
- The interpreter runs the input through the pre-trained model.
- The output includes:
- Predicted Task Scores: Confidence scores for each possible task.
- Selected Task: The task with the highest score.


### **How the AI Adapts in Real-Time**

- After completing a task, the robot feeds its current state (task + sensor data) into the model.
- The AI processes the input and:
- Predicts the next task to perform.
- Scores all potential tasks to indicate confidence levels.
- The robot executes the predicted task with the highest score.


# ATEM: Adaptive Task Execution and Machine Learning Package Documentation

ATEM is a Python package designed for adaptive task execution in robotics and AI applications. It provides tools for training machine learning models, interpreting task sequences, and generating optimal task orders for various scenarios.

## Features
- **Adaptive Task Model**: Predict the next task based on sensor data and task history.
- **Task Training**: Train custom machine learning models using a `tasks.json` file.
- **Real-time Adaptation**: Simulate real-world scenarios for task execution.
- **Pathfinding Integration**: Extendable for integration with A* pathfinding for robotics.
- **Lightweight TensorFlow Lite Integration**: For efficient model inference.

---

## Installation

```bash
pip install atem
```
---

## Quick Start Guide
1. **Preparing the tasks.json File**
The tasks.json file defines the tasks and their attributes.

Example
```json
{
    "tasks": [
        {"name": "Task 1", "points": 10, "time": 5},
        {"name": "Task 2", "points": 20, "time": 15},
        {"name": "Task 3", "points": 15, "time": 10}
    ]
}
```
2. **Training a Model**
Use the ModelTrainer class to train a TensorFlow Lite model.

Example
```python
from atem.model_train import ModelTrainer

trainer = ModelTrainer(tasks_file="tasks.json", output_model_path="adaptive_model.tflite")
trainer.train_and_save_model(epochs=20, batch_size=16)

```

3. **Interpreting Tasks**
Use the AdaptiveModel class to interpret task sequences and predict the next task.

Example
```python
from atem import AdaptiveModel

model = AdaptiveModel(model_path="adaptive_model.tflite")

task_to_index = {"Task 1": 0, "Task 2": 1, "Task 3": 2}
index_to_task = {0: "Task 1", 1: "Task 2", 2: "Task 3"}
current_task = "Task 1"
sensor_data = {
    "time_elapsed": 20,
    "distance_to_target": 1.2,
    "gyro_angle": 45,
    "battery_level": 80
}
predicted_task, scores = model.predict_next_task(
    current_task=current_task,
    sensor_data=sensor_data,
    task_to_index=task_to_index,
    index_to_task=index_to_task,
    max_length=5
)

print(f"Predicted Next Task: {predicted_task}")
print(f"Task Scores: {scores}")

```

---

## API Reference

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














