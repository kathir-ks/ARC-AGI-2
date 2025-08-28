import json
import os
import argparse
from collections import deque
from PIL import Image, ImageDraw

def find_objects(grid: list[list[int]], background_color: int = 0) -> list[dict]:
    """
    Finds all distinct objects in a grid based on color and 8-directional connectivity.
    """
    if not grid or not grid[0]:
        return []

    height = len(grid)
    width = len(grid[0])
    visited = [[False for _ in range(width)] for _ in range(height)]
    found_objects = []

    directions = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
    ]

    for r in range(height):
        for c in range(width):
            if grid[r][c] == background_color or visited[r][c]:
                continue

            color = grid[r][c]
            current_object_cells = []
            min_r, max_r = r, r
            min_c, max_c = c, c

            queue = deque([(r, c)])
            visited[r][c] = True

            while queue:
                curr_r, curr_c = queue.popleft()
                current_object_cells.append((curr_r, curr_c))

                min_r, max_r = min(min_r, curr_r), max(max_r, curr_r)
                min_c, max_c = min(min_c, curr_c), max(max_c, curr_c)

                for dr, dc in directions:
                    nr, nc = curr_r + dr, curr_c + dc
                    if (0 <= nr < height and 0 <= nc < width and
                            not visited[nr][nc] and grid[nr][nc] == color):
                        visited[nr][nc] = True
                        queue.append((nr, nc))
            
            # Sort cells to have a consistent order, though not strictly necessary
            # for the signature generation which also sorts.
            current_object_cells.sort()
            found_objects.append({
                'color': color,
                'cells': current_object_cells,
                'bounding_box': (min_r, min_c, max_r, max_c)
            })
    return found_objects

def visualize_objects(grid: list[list[int]], objects: list[dict], filename: str):
    """
    Creates and saves an image visualizing the grid and the bounding boxes of found objects.
    """
    cell_size = 20
    height = len(grid)
    width = len(grid[0])
    
    img_width = width * cell_size
    img_height = height * cell_size

    # Base color map for grid cells
    grid_color_map = {
        0: (255, 255, 255), 1: (255, 0, 0),     2: (0, 0, 255),
        3: (0, 128, 0),     4: (255, 255, 0),   5: (128, 0, 128),
        6: (255, 165, 0),   7: (255, 192, 203), 8: (128, 128, 128),
        9: (165, 42, 42)
    }

    # Bright, distinct colors for bounding boxes
    box_colors = [
        (3, 252, 240), (252, 3, 161), (3, 252, 11), (252, 148, 3),
        (148, 3, 252), (3, 78, 252)
    ]

    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw the grid cells
    for r in range(height):
        for c in range(width):
            color = grid_color_map.get(grid[r][c], (200, 200, 200))
            x0, y0 = c * cell_size, r * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(50, 50, 50))

    # Draw the bounding boxes and labels for each object
    for i, obj in enumerate(objects):
        min_r, min_c, max_r, max_c = obj['bounding_box']
        
        # Bounding box coordinates
        x0 = min_c * cell_size - 1
        y0 = min_r * cell_size - 1
        x1 = (max_c + 1) * cell_size
        y1 = (max_r + 1) * cell_size

        box_color = box_colors[i % len(box_colors)]
        draw.rectangle([x0, y0, x1, y1], outline=box_color, width=2)
        
        # Add a label
        label = f"{i+1}"
        draw.text((x0 + 3, y0 + 1), label, fill=box_color)

    img.save(filename)
    print(f"Saved visualization to {filename}")

def print_objects(objects: list[dict], grid_name: str):
    """Helper function to print found objects in a readable format."""
    print(f"--- Objects found in {grid_name} ---")
    if not objects:
        print("No objects found.")
        return
        
    for i, obj in enumerate(objects):
        print(f"Object {i+1}:")
        print(f"  - Color: {obj['color']}")
        print(f"  - Cell Count: {len(obj['cells'])}")
        print(f"  - Bounding Box (min_r, min_c, max_r, max_c): {obj['bounding_box']}")
    print("-" * (len(grid_name) + 24))

def get_object_signature(obj: dict) -> tuple:
    """
    Generates a position-independent signature for an object based on its color and shape.
    The shape is represented by normalizing the cell coordinates relative to the top-leftmost cell.
    """
    color = obj['color']
    cells = obj['cells']
    
    if not cells:
        return (color, tuple())

    # Find the top-leftmost coordinate of the object
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    
    # Normalize coordinates by shifting the object so its top-left point is at (0, 0)
    normalized_cells = sorted([(r - min_r, c - min_c) for r, c in cells])
    
    # The signature is a tuple of the color and the tuple of normalized cell coordinates
    return (color, tuple(normalized_cells))

def find_and_print_similarities(input_objects: list[dict], output_objects: list[dict]):
    """
    Finds and prints pairs of similar objects between two lists of objects.
    Similarity is determined by matching object signatures. This function handles
    multiple identical objects by matching them on a one-to-one basis.
    """
    print("\n--- Similarity Analysis ---")
    
    # Generate signatures for all objects
    input_signatures = [get_object_signature(obj) for obj in input_objects]
    output_signatures = [get_object_signature(obj) for obj in output_objects]

    matches = []
    # Keep track of output objects that have already been matched
    used_output_indices = [False] * len(output_objects)

    # Find pairs with matching signatures
    for i, in_sig in enumerate(input_signatures):
        for j, out_sig in enumerate(output_signatures):
            # If signatures match and the output object hasn't been matched yet
            if in_sig == out_sig and not used_output_indices[j]:
                matches.append({'input_index': i + 1, 'output_index': j + 1})
                used_output_indices[j] = True # Mark this output object as used
                break # Move to the next input object

    if not matches:
        print("No similar objects found between input and output.")
    else:
        print(f"Found {len(matches)} pair(s) of similar objects:")
        for match in matches:
            in_idx = match['input_index']
            out_idx = match['output_index']
            # Retrieve color and cell count for the report
            color = input_objects[in_idx - 1]['color']
            cell_count = len(input_objects[in_idx - 1]['cells'])
            print(
                f"  - Input Object #{in_idx} matches Output Object #{out_idx} "
                f"(Color: {color}, Cells: {cell_count})"
            )
    print("---------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and visualize objects in ARC-AGI grids."
    )
    # Add an argument to disable visualization; it's enabled by default
    parser.add_argument(
        "--no-visualization",
        dest="visualize",
        action="store_false",
        help="Disable the generation of visualization images."
    )
    # Add an argument for the output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization images."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist and visualization is on
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Example Usage ---
    example_task = {
        "train": [{
            "input": [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 2, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 2, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 2, 2, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        }]
    }

    input_grid = example_task["train"][0]["input"]
    output_grid = example_task["train"][0]["output"]

    # --- Process Input Grid ---
    input_objects = find_objects(input_grid)
    print_objects(input_objects, "Input Grid")
    if args.visualize:
        viz_path = os.path.join(args.output_dir, "input_grid_viz.png")
        visualize_objects(input_grid, input_objects, viz_path)

    # --- Process Output Grid ---
    output_objects = find_objects(output_grid)
    print_objects(output_objects, "Output Grid")
    if args.visualize:
        viz_path = os.path.join(args.output_dir, "output_grid_viz.png")
        visualize_objects(output_grid, output_objects, viz_path)
    
    # --- Find and Print Similarities ---
    find_and_print_similarities(input_objects, output_objects)