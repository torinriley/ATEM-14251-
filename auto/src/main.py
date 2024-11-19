import heapq
import json
from auto.src.a_star import AStar
from auto.src.simulation import RobotSimulator


def main():
    # Load field configuration
    field_config = {
        "field_width": 6,
        "field_height": 6,
        "non_traversable": [[2, 2], [2, 3], [3, 2], [3, 3]],
        "points_of_interest": {
            "start": [0, 0],
            "goal": [5, 5]
        }
    }

    # Initialize A* and robot simulator
    astar = AStar(field_config)
    robot = RobotSimulator(grid_width=field_config["field_width"],
                           grid_height=field_config["field_height"],
                           start_position=field_config["points_of_interest"]["start"])

    # Perform A* search
    start = tuple(field_config["points_of_interest"]["start"])
    goal = tuple(field_config["points_of_interest"]["goal"])
    path = astar.a_star_search(start, goal)

    if path:
        print(f"Path found: {path}")
        robot.execute_path(path)
    else:
        print("No path found.")


if __name__ == "__main__":
    main()