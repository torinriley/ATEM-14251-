import json
import random

OUTPUT_JSON_PATH = "tasks.json"

# Define task templates
TASK_NAMES = [
    "Observation Zone", "Net Zone", "Low Basket", "High Basket",
    "Low Chamber", "High Chamber", "Level 1 Ascent", "Level 2 Ascent", "Level 3 Ascent"
]

# Define point ranges and time ranges for each task type
TASK_PROPERTIES = {
    "Observation Zone": {"points": (1, 3), "time": (2, 5)},
    "Net Zone": {"points": (2, 4), "time": (3, 6)},
    "Low Basket": {"points": (4, 8), "time": (4, 8)},
    "High Basket": {"points": (6, 10), "time": (6, 10)},
    "Low Chamber": {"points": (8, 12), "time": (6, 12)},
    "High Chamber": {"points": (10, 15), "time": (10, 15)},
    "Level 1 Ascent": {"points": (3, 5), "time": (10, 15)},
    "Level 2 Ascent": {"points": (10, 20), "time": (15, 20)},
    "Level 3 Ascent": {"points": (25, 30), "time": (20, 30)}
}

def generate_task_data(num_tasks=50):
    tasks = []
    for _ in range(num_tasks):
        task_name = random.choice(TASK_NAMES)
        task = {
            "name": task_name,
            "points": random.randint(*TASK_PROPERTIES[task_name]["points"]),
            "time": random.randint(*TASK_PROPERTIES[task_name]["time"])
        }
        tasks.append(task)
    return tasks

def save_task_data(tasks, output_path):
    """Save tasks to a JSON file."""
    data = {"tasks": tasks}
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Task data saved to {output_path}")

if __name__ == "__main__":
    # Generate tasks and save to file
    print("Generating task data...")
    tasks = generate_task_data(num_tasks=50)
    save_task_data(tasks, OUTPUT_JSON_PATH)
    print("Task generation complete.")