import numpy as np
import cv2

class AvoidObstacle:
    def __init__(self, grid_size=160, resolution=0.025, radius=2.0, safe_margin=1.0):
        self.grid_size = grid_size
        self.resolution = resolution
        self.radius = radius
        self.safe_margin = safe_margin

        if self.radius == 2.0:
            self.search_angle = 29.0
            self.search_times = 2
        elif self.radius == 6.0:
            self.search_angle = 10.0
            self.search_times = 6



    def create_obstacle_map(self, point_cloud):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for point in point_cloud:
            x, y, z = point
            if z < self.radius:  # Use z for depth thresholding as before
                # Convert x and z to grid coordinates
                grid_x = int((x / self.resolution) + (self.grid_size / 2))
                grid_z = int((z / self.resolution) + (self.grid_size / 2))
                if 0 <= grid_x < self.grid_size and 0 <= grid_z < self.grid_size:
                    grid[grid_x][grid_z] = 255  # Mark as obstacle

        # # Center of the grid, assuming the sensor is at the center
        # center_x, center_z = self.grid_size // 2, self.grid_size // 2
        # radius = int(0.5 / self.resolution)  # 0.5 m radius to grid cells

        # # Draw the circle in the grid
        # cv2.circle(grid, (center_x, center_z), radius, (128), 1)

        center_x, center_z = self.grid_size // 2, self.grid_size // 2
        radius = int(self.radius / self.resolution)  # 0.5 m radius to grid cells

        cv2.circle(grid, (center_x, center_z), radius, (128), 1)
        angle = 14.5
        angle_rad = np.radians(angle)
        x1 = center_x
        y1 = center_z
        x2 = center_x + int(radius * np.cos(angle_rad))
        y2 = center_z + int(radius * np.sin(angle_rad))
        x3 = center_x + int(radius * np.cos(-angle_rad))
        y3 = center_z + int(radius * np.sin(-angle_rad))
        cv2.line(grid, (x1, y1), (x2, y2), (128), 1)
        cv2.line(grid, (x1, y1), (x3, y3), (128), 1)

        # rotate the image ccw by 90 degrees
        # only keep the top half of the image
        grid = np.rot90(grid)
        grid = grid[:self.grid_size//2, :]

        self.obs_map = grid
        self.width, self.height = grid.shape
        self.pivot_abs = (int(self.width / 2), int(self.height - 1))
    
    def check_free_space(self, image):
        x1, y1, x2, y2 = self.pivot_abs[0]-self.safe_margin/(2*self.resolution), self.pivot_abs[1]-self.radius/self.resolution, self.pivot_abs[0]+self.safe_margin/(2*self.resolution), self.pivot_abs[1]

        # Extract the region of the rectangle
        region = image.crop((x1, y1, x2, y2))

        # Convert the region to a numpy array for easier manipulation
        region_array = np.array(region)

        # Check if there is any 255 in the rectangle
        has_255 = 255 in region_array

        return has_255
    
    def rotate_image(self, angle):
        # mask = Image.fromarray(mask)
        # Calculate the pivot point coordinates relative to the image dimensions
        

        # Perform the rotation
        rotated_image = self.obs_map.rotate(angle, center=self.pivot_abs, expand=False)

        # Save the rotated image
        return rotated_image
    
    def find_g1(self, current_position, current_heading, goal_position):
        if self.check_free_space(self.obs_map):
            return goal_position, current_heading
        else:
            distance = np.linalg.norm(np.array(current_position) - np.array(goal_position))
            for i in range(self.search_times):
                angle = self.search_angle * (i + 1)
                rotated_image = self.rotate_image(angle)
                if self.check_free_space(rotated_image):
                    goal_heading = current_heading + angle
                    goal_position_x = distance*np.cos(goal_heading) + current_position[0]
                    goal_position_y = distance*np.sin(goal_heading) + current_position[1]
                    goal_position = (goal_position_x, goal_position_y)
                    return goal_position, goal_heading
                else:
                    angle = -self.search_angle * (i + 1)
                    rotated_image = self.rotate_image(angle)
                    if self.check_free_space(rotated_image):
                        goal_heading = current_heading + angle
                        goal_position_x = distance*np.cos(goal_heading) + current_position[0]
                        goal_position_y = distance*np.sin(goal_heading) + current_position[1]
                        goal_position = (goal_position_x, goal_position_y)
                        return goal_position, goal_heading
                    
            return current_position, current_heading
            



