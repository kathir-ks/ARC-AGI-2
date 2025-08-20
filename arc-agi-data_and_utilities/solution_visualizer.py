import json
import os
from PIL import Image, ImageDraw
import argparse

class SolutionImageGenerator:
    """
    Generates images from ARC competition solution grids stored in a JSON file.
    The expected JSON format is a dictionary mapping problem IDs to solution grids:
    {
        "problem_id_1": [[[...grid...]]],
        "problem_id_2": [[[...grid...]]],
        ...
    }
    """
    def __init__(self, json_file_path, output_dir="arc_solutions_output"):
        """
        Initializes the generator.

        Args:
            json_file_path (str): The path to the input solutions JSON file.
            output_dir (str): The directory where output images will be saved.
        """
        self.json_file_path = json_file_path
        self.output_dir = output_dir
        self.data = None
        # Standard ARC color map
        self.color_map = {
            0: (0, 0, 0),       # black (used for outlines, not usually a grid color)
            1: (0, 0, 255),     # blue
            2: (255, 0, 0),     # red
            3: (0, 255, 0),     # green
            4: (255, 255, 0),   # yellow
            5: (128, 128, 128), # gray
            6: (255, 165, 0),   # orange -> fuchsia/magenta in official viewer
            7: (255, 192, 203), # pink -> orange in official viewer
            8: (0, 128, 0),     # dark green -> teal in official viewer
            9: (165, 42, 42)    # brown
        }
        # A background color for padding, distinct from grid colors
        self.background_color = (255, 255, 255) # white

        self.load_data()
        self.create_output_directory()

    def load_data(self):
        """Loads the JSON data from the specified file path."""
        try:
            with open(self.json_file_path, 'r') as file:
                self.data = json.load(file)
            print(f"Successfully loaded data from {self.json_file_path}")
        except FileNotFoundError:
            print(f"Error: File not found at {self.json_file_path}")
            self.data = None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.json_file_path}: {e}")
            self.data = None

    def create_output_directory(self):
        """Creates the output directory if it doesn't already exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def create_grid_image(self, grid_data, cell_size=20):
        """
        Creates a single image from a grid data array.
        
        Args:
            grid_data (list[list[int]]): The 2D list representing the grid.
            cell_size (int): The size of each square cell in pixels.

        Returns:
            PIL.Image.Image: The generated image object, or None if input is invalid.
        """
        if not grid_data or not isinstance(grid_data, list) or not isinstance(grid_data[0], list):
            print("Warning: Invalid grid data format. Skipping image creation.")
            return None

        rows = len(grid_data)
        cols = len(grid_data[0])
        
        img_width = cols * cell_size
        img_height = rows * cell_size
        
        img = Image.new('RGB', (img_width, img_height), color=self.background_color)
        draw = ImageDraw.Draw(img)
        
        # Draw the grid cells
        for r in range(rows):
            for c in range(cols):
                value = grid_data[r][c]
                # Use a default color (light gray) if the value is not in the map
                rgb_color = self.color_map.get(value, (200, 200, 200)) 
                
                x0 = c * cell_size
                y0 = r * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                
                # Draw the rectangle for the cell
                draw.rectangle([x0, y0, x1, y1], fill=rgb_color, outline=(0,0,0))
        
        return img

    def generate_all_solution_images(self):
        """Iterates through the loaded data and saves an image for each solution."""
        if not self.data:
            print("No data loaded. Cannot generate images.")
            return

        print("Starting image generation...")
        for problem_key, solution_data in self.data.items():
            # The JSON format nests the grid inside a list: [[...grid...]]
            if solution_data and isinstance(solution_data, list) and len(solution_data) > 0:
                # Extract the actual grid which is the first element
                output_grid = solution_data[0]
                
                # Create the image for this grid
                output_img = self.create_grid_image(output_grid)
                
                if output_img:
                    filename = f"{problem_key}_solution.png"
                    file_path = os.path.join(self.output_dir, filename)
                    output_img.save(file_path)
                    print(f"Saved solution image: {file_path}")
            else:
                print(f"Skipping '{problem_key}': No valid solution data found.")
        print("Image generation complete.")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate images for ARC competition solution grids from a JSON file."
    )
    parser.add_argument(
        "--file", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing the solution grids."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="arc_solutions_output",
        help="Directory to save the generated images."
    )
    args = parser.parse_args()
    
    # Create an instance of the generator and run it
    image_gen = SolutionImageGenerator(json_file_path=args.file, output_dir=args.output)
    image_gen.generate_all_solution_images()
