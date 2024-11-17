import json
from a_star import AStar

def main():
    with open("field_config.json", "r") as file:
        field_config = json.load(file)

    a_star = AStar(field_config)

    start = (0, 0)
    goal = tuple(field_config["points_of_interest"]["blue_scoring_zone"])

    path = a_star.a_star_search(start, goal)

    if path:
        print("Path found:", path)
    else:
        print("No path found!")

if __name__ == "__main__":
    main()