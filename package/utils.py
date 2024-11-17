import json

def load_tasks(file_path):
    with open(file_path, "r") as file:
        return json.load(file)["tasks"]

def create_task_encoder(tasks):
    task_names = sorted(set(task["name"] for task in tasks))
    task_to_index = {name: idx for idx, name in enumerate(task_names)}
    index_to_task = {idx: name for name, idx in task_to_index.items()}
    return task_to_index, index_to_task