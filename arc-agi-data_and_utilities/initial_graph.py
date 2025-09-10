import json
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import deque, defaultdict

# --- Data Structures for Graph Representation (largely unchanged) ---

class ArcObject:
    """Represents a single connected object in the grid."""
    def __init__(self, obj_id: int, color: int, pixels: Set[Tuple[int, int]]):
        self.id = obj_id
        self.color = color
        self.pixels = pixels
        self.size = len(pixels)
        self.bounding_box = self._calculate_bounding_box()
        self.centroid = self._calculate_centroid()
        self.shape_hash = self._calculate_shape_hash()

    def _calculate_bounding_box(self) -> Tuple[int, int, int, int]:
        min_r = min(r for r, c in self.pixels)
        max_r = max(r for r, c in self.pixels)
        min_c = min(c for r, c in self.pixels)
        max_c = max(c for r, c in self.pixels)
        return (min_r, min_c, max_r, max_c)

    def _calculate_centroid(self) -> Tuple[float, float]:
        sum_r = sum(r for r, c in self.pixels)
        sum_c = sum(c for r, c in self.pixels)
        return (sum_r / self.size, sum_c / self.size)
        
    def _calculate_shape_hash(self) -> Tuple[Tuple[int, int], ...]:
        min_r, min_c, _, _ = self.bounding_box
        normalized_pixels = sorted([(r - min_r, c - min_c) for r, c in self.pixels])
        return tuple(normalized_pixels)

    def __repr__(self) -> str:
        return f"Obj(id={self.id}, c={self.color}, sz={self.size}, pos=({self.bounding_box[0]},{self.bounding_box[1]}))"

class SceneGraph:
    """Represents a grid as a graph of objects and their relationships."""
    def __init__(self):
        self.objects: Dict[int, ArcObject] = {}
        self.adj: Dict[int, List[Tuple[str, int]]] = {}

    def add_object(self, obj: ArcObject):
        self.objects[obj.id] = obj
        self.adj[obj.id] = []

    def add_edge(self, obj1_id: int, obj2_id: int, relation: str):
        self.adj[obj1_id].append((relation, obj2_id))

    def __repr__(self):
        return f"Graph with {len(self.objects)} objects."

# --- Core Logic Functions ---

def find_objects(grid: np.ndarray) -> List[ArcObject]:
    """Identifies all connected objects in a grid using BFS."""
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    objects = []
    obj_id_counter = 0
    for r in range(height):
        for c in range(width):
            if not visited[r, c] and grid[r, c] != 0:
                color = grid[r, c]
                pixels: Set[Tuple[int, int]] = set()
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    curr_r, curr_c = q.popleft()
                    pixels.add((curr_r, curr_c))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (0 <= nr < height and 0 <= nc < width and
                                not visited[nr, nc] and grid[nr, nc] == color):
                            visited[nr, nc] = True
                            q.append((nr, nc))
                objects.append(ArcObject(obj_id_counter, color, pixels))
                obj_id_counter += 1
    return objects

def build_scene_graph(grid: np.ndarray) -> SceneGraph:
    """Constructs a scene graph from a raw grid."""
    objects = find_objects(grid)
    graph = SceneGraph()
    for obj in objects:
        graph.add_object(obj)
    # Edge building logic can be expanded here
    return graph

def find_transformations(input_graph: SceneGraph, output_graph: SceneGraph) -> List[Dict]:
    """Compares graphs to find specific transformations for each object."""
    transformations = []
    unmatched_output_objs = list(output_graph.objects.values())

    for in_obj in input_graph.objects.values():
        best_match = None
        # Find best match based on shape, then color
        for out_obj in unmatched_output_objs:
            if in_obj.shape_hash == out_obj.shape_hash:
                best_match = out_obj
                if in_obj.color == out_obj.color:
                    break # Perfect match
        
        if best_match:
            unmatched_output_objs.remove(best_match)
            trans_info = {'input_obj': in_obj, 'output_obj': best_match}
            
            # Check for movement
            translation = (
                round(best_match.centroid[0] - in_obj.centroid[0]),
                round(best_match.centroid[1] - in_obj.centroid[1])
            )
            if translation != (0, 0):
                trans_info['type'] = 'move'
                trans_info['translation'] = translation
            
            # Check for color change
            if in_obj.color != best_match.color:
                trans_info['type'] = 'recolor'
                trans_info['color_map'] = (in_obj.color, best_match.color)
            
            if 'type' in trans_info:
                transformations.append(trans_info)
    
    return transformations

def generalize_rule(transformations: List[Dict], input_graph: SceneGraph) -> Optional[Dict]:
    """Analyzes transformations to induce a single general rule."""
    if not transformations:
        return None

    # Simple generalization: find the most common transformation type and its properties
    move_translations = defaultdict(list)
    for t in transformations:
        if t['type'] == 'move':
            # Group translations by the color of the object being moved
            color = t['input_obj'].color
            move_translations[color].append(t['translation'])
    
    # Check if all objects of a certain color moved by the same amount
    for color, translations in move_translations.items():
        if len(set(translations)) == 1:
            # All objects of this color moved consistently. Assume this is the rule.
            return {
                'type': 'conditional_move',
                'condition': {'color': color},
                'action': {'translation': translations[0]}
            }
            
    # Add more generalization logic here (e.g., for recoloring, deletion, etc.)
    return None

def apply_rule_and_reconstruct(input_grid: np.ndarray, rule: Dict) -> np.ndarray:
    """Applies a generalized rule to an input grid to produce an output grid."""
    if not rule:
        return np.copy(input_grid) # Return original if no rule found

    input_graph = build_scene_graph(input_grid)
    output_objects = []

    for obj in input_graph.objects.values():
        transformed_pixels = obj.pixels
        transformed_color = obj.color

        # Check if the rule's condition matches the object
        if rule['type'] == 'conditional_move':
            if obj.color == rule['condition']['color']:
                # Apply the action
                dr, dc = rule['action']['translation']
                transformed_pixels = set((r + dr, c + dc) for r, c in obj.pixels)
        
        # Create a new object with the transformed properties
        # NOTE: ID doesn't matter for reconstruction
        output_objects.append(ArcObject(obj.id, transformed_color, transformed_pixels))

    # Reconstruct the grid from the new list of objects
    output_grid = np.zeros_like(input_grid)
    for obj in output_objects:
        for r, c in obj.pixels:
            if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                output_grid[r, c] = obj.color
                
    return output_grid

# --- Main Execution ---

def print_grid(grid: np.ndarray, title: str):
    print(f"--- {title} ---")
    if grid is None: 
        print("None")
        return
    for row in grid:
        print(" ".join(map(str, row)))
    print("-" * 25)

def main():
    """Main function to load a task, solve it, and display results."""
    challenges_file = 'arc-agi_training_challenges.json'
    solutions_file = 'arc-agi_training_solutions.json'

    # Task '6150a2bd' involves moving objects of a specific color (blue=2).
    task_id = '6150a2bd' 

    try:
        with open(challenges_file, 'r') as f: challenges = json.load(f)
        with open(solutions_file, 'r') as f: solutions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ensure '{challenges_file}' and '{solutions_file}' are present.")
        return

    task = challenges.get(task_id)
    task_solutions = solutions.get(task_id)
    if not task or not task_solutions:
        print(f"Task '{task_id}' not found.")
        return
        
    print(f"--- Analyzing Task: {task_id} ---\n")
    # 1. Learn from the first training example
    train_pair = task['train'][0]
    input_grid = np.array(train_pair['input'])
    output_grid = np.array(train_pair['output'])

    input_graph = build_scene_graph(input_grid)
    output_graph = build_scene_graph(output_grid)
    
    transformations = find_transformations(input_graph, output_graph)
    print(f"Found {len(transformations)} specific transformations.")
    
    rule = generalize_rule(transformations, input_graph)
    print(f"Generalized Rule: {rule}\n")

    # 2. Apply the learned rule to the first test case
    print("--- Running Inference on Test Case ---")
    test_input_grid = np.array(task['test'][0]['input'])
    actual_solution_grid = np.array(task_solutions[0])

    predicted_grid = apply_rule_and_reconstruct(test_input_grid, rule)

    # 3. Display results
    print_grid(test_input_grid, "Test Input")
    print_grid(predicted_grid, "Predicted Output")
    print_grid(actual_solution_grid, "Actual Solution")
    
    if np.array_equal(predicted_grid, actual_solution_grid):
        print("\n✅ Success! The prediction matches the solution.")
    else:
        print("\n❌ Failure. The prediction does not match the solution.")


if __name__ == '__main__':
    main()

