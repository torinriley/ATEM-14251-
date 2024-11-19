
class RobotSimulator:
    def __init__(self, grid_width, grid_height, start_position=(0, 0), start_orientation="NORTH"):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.position = list(start_position)
        self.orientation = start_orientation
        self.valid_orientations = ["NORTH", "EAST", "SOUTH", "WEST"]

    def move_to(self, next_position):
        current_x, current_y = self.position
        next_x, next_y = next_position

        if next_x > current_x:
            self.set_orientation("EAST")
        elif next_x < current_x:
            self.set_orientation("WEST")
        elif next_y > current_y:
            self.set_orientation("NORTH")
        elif next_y < current_y:
            self.set_orientation("SOUTH")

        self.position = [next_x, next_y]
        print(f"Moved to: {self.position}, Orientation: {self.orientation}")

    def set_orientation(self, new_orientation):
        self.orientation = new_orientation
        print(f"Orientation set to: {self.orientation}")

    def execute_path(self, path):
        for next_position in path:
            self.move_to(next_position)

    def get_position(self):
        return tuple(self.position)

    def get_orientation(self):
        return self.orientation


