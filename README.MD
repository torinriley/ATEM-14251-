





<img alt="atem" src="other/atem.png" width="600" height="200" style="padding: 20px; display: block; margin-left: auto; margin-right: auto;"/>


# ATEM - Adaptive Task Execution Model

---
**ATEM** (Adaptive Task Execution Model) is a machine learning-based project designed to determine the optimal sequence of tasks to maximize points in the autonomous phase of FTC robotics competitions. The project utilizes TensorFlow and TensorFlow Lite for efficient model deployment.



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
