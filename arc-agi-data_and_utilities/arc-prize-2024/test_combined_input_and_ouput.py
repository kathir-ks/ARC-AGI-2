import json
import os
from PIL import Image, ImageDraw
import argparse

class ImageGenerator:
    """
    Generates images from grid data defined in ARC JSON files.
    It can create combined images for input-output pairs for both training
    and testing datasets.
    """
    def __init__(self, json_file_path, test_output_json_path=None, output_dir="arc_images_solutions_combined"):
        """
        Initializes the ImageGenerator.

        Args:
            json_file_path (str): Path to the main JSON file with training and test inputs.
            test_output_json_path (str, optional): Path to the JSON file with test outputs. Defaults to None.
            output_dir (str, optional): Directory to save the generated images. Defaults to "arc_images".
        """
        self.json_file_path = json_file_path
        self.test_output_json_path = test_output_json_path # New: Path for test outputs
        self.output_dir = output_dir
        self.data = None
        self.test_outputs = None # New: To store loaded test outputs
        self.color_map = {
            0: (255, 255, 255),       # black (changed from white for better visibility on white background)
            1: (255, 0, 0),     # red
            2: (0, 0, 255),     # blue
            3: (0, 128, 0),     # green
            4: (255, 255, 0),   # yellow
            5: (128, 0, 128),   # purple
            6: (255, 165, 0),   # orange
            7: (255, 192, 203), # pink
            8: (128, 128, 128), # gray
            9: (165, 42, 42)    # brown
        }
        
        self.load_data()
        self.load_test_outputs() # New: Load the test output data
        self.create_output_directory()

    def load_data(self):
        """Loads the main JSON data file."""
        try:
            with open(self.json_file_path, 'r') as file:
                self.data = json.load(file)
            print(f"Loaded data from {self.json_file_path}")
        except FileNotFoundError:
            print(f"Error: File not found at {self.json_file_path}")
            self.data = None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.json_file_path}: {e}")
            self.data = None

    def load_test_outputs(self):
        """New: Loads the JSON file containing test outputs if a path is provided."""
        if not self.test_output_json_path:
            print("No test output file provided. Test images will only show inputs.")
            return
        
        try:
            with open(self.test_output_json_path, 'r') as file:
                self.test_outputs = json.load(file)
            print(f"Loaded test outputs from {self.test_output_json_path}")
        except FileNotFoundError:
            print(f"Error: Test output file not found at {self.test_output_json_path}")
            self.test_outputs = None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.test_output_json_path}: {e}")
            self.test_outputs = None

    def create_output_directory(self):
        """Creates the output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")

    def create_padded_grid_image(self, grid_data, max_size=30, cell_size=20):
        """Creates a single grid image with padding."""
        if not grid_data or not isinstance(grid_data, list) or not isinstance(grid_data[0], list):
            print(f"Warning: Skipping invalid grid data: {grid_data}")
            return None

        rows = len(grid_data)
        cols = len(grid_data[0])
        
        # Create a new image with a white background for padding
        padded_width = max(cols, max_size)
        padded_height = max(rows, max_size)
        
        img_width = padded_width * cell_size
        img_height = padded_height * cell_size
        
        img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Calculate padding to center the grid
        pad_x = (padded_width - cols) // 2
        pad_y = (padded_height - rows) // 2

        # Draw the grid cells
        for r in range(rows):
            for c in range(cols):
                value = grid_data[r][c]
                rgb_color = self.color_map.get(value, (200, 200, 200)) # lightgray default
                
                # Calculate coordinates for the padded grid
                x0 = (c + pad_x) * cell_size
                y0 = (r + pad_y) * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                
                draw.rectangle([x0, y0, x1, y1], fill=rgb_color, outline=(0,0,0)) # Added outline for grid lines
        
        return img

    def save_combined_image(self, input_grid, output_grid, filename, max_size=30, cell_size=20):
        """Creates and saves a side-by-side image of an input and output grid."""
        input_img = self.create_padded_grid_image(input_grid, max_size, cell_size)
        output_img = self.create_padded_grid_image(output_grid, max_size, cell_size)
        
        if not input_img or not output_img:
            print(f"Skipping combined image for {filename} due to missing data.")
            return

        # Create a larger canvas to fit both images side by side with a margin
        total_width = input_img.width + output_img.width + cell_size 
        total_height = max(input_img.height, output_img.height)
        
        combined_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
        
        # Paste the input and output images onto the new canvas
        combined_img.paste(input_img, (0, 0))
        combined_img.paste(output_img, (input_img.width + cell_size, 0))
        
        file_path = os.path.join(self.output_dir, filename)
        combined_img.save(file_path)
        print(f"Saved combined image: {file_path}")

    def generate_all_images(self):
        """Generates all images for the loaded training and test data."""
        if not self.data:
            print("No data loaded. Exiting image generation.")
            return

        for problem_key, problem_data in self.data.items():
            train_data = problem_data.get('train', [])
            test_data = problem_data.get('test', [])

            # Process training data (always has input and output)
            for i, pair in enumerate(train_data):
                input_grid = pair.get('input', [])
                output_grid = pair.get('output', [])
                self.save_combined_image(input_grid, output_grid, f"{problem_key}_train_{i}_combined.png")

            # Process test data
            for i, pair in enumerate(test_data):
                input_grid = pair.get('input', [])
                output_grid = None # Assume no output grid initially

                # New: Try to find the corresponding test output
                if self.test_outputs and problem_key in self.test_outputs:
                    if i < len(self.test_outputs[problem_key]):
                        # *** FIX: Access the entire grid, not just the first row ***
                        output_grid = self.test_outputs[problem_key][i]
                    else:
                        print(f"Warning: Output for {problem_key}_test_{i} not found in test output file.")

                # If an output grid was found, create a combined image
                if output_grid:
                    self.save_combined_image(input_grid, output_grid, f"{problem_key}_test_{i}_combined.png")
                # Otherwise, fall back to saving only the input image
                else:
                    input_img = self.create_padded_grid_image(input_grid)
                    if input_img:
                        file_path = os.path.join(self.output_dir, f"{problem_key}_test_{i}_input.png")
                        input_img.save(file_path)
                        print(f"Saved test input image: {file_path}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from ARC JSON files.")
    parser.add_argument("--file", type=str, required=True, help="Path to the main JSON file (e.g., training data).")
    # New: Added argument for the test output file
    parser.add_argument("--test_outputs", type=str, help="Optional: Path to the JSON file containing test outputs.")
    
    args = parser.parse_args()

    # Pass both file paths to the generator
    image_gen = ImageGenerator(json_file_path=args.file, test_output_json_path=args.test_outputs)
    image_gen.generate_all_images()
