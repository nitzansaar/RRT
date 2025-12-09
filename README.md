# RRT (Rapidly-exploring Random Tree) Visualization

A Python implementation of the RRT path planning algorithm with real-time visualization.


https://github.com/user-attachments/assets/c8ed1382-fc40-4ed7-8092-7aeb137087ef



## Features

- **Interactive GUI**: Visualizes the RRT tree as it grows in real-time
- **Path Planning**: Finds a collision-free path from start to goal
- **Obstacle Avoidance**: Supports circular and rectangular obstacles
- **Path Highlighting**: Automatically highlights the shortest path once the goal is reached

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the visualization:
```bash
python rrt_visualization.py
```

## How It Works

1. **Tree Growth**: The algorithm starts from the start position and randomly samples points in the workspace
2. **Nearest Node**: For each random sample, it finds the nearest existing node in the tree
3. **Steering**: It creates a new node by steering from the nearest node towards the random sample (with a maximum step size)
4. **Collision Checking**: Before adding a new node, it checks if the path is collision-free
5. **Goal Connection**: When a node gets close enough to the goal, it attempts to connect directly to the goal
6. **Path Extraction**: Once the goal is reached, it backtracks from the goal to the start to find the path

## Customization

You can customize the following parameters in `main()`:

- **Bounds**: Workspace boundaries `(x_min, x_max, y_min, y_max)`
- **Start/Goal**: Start and goal positions `(x, y)`
- **Obstacles**: List of obstacles (circular or rectangular)
- **Step Size**: Maximum distance for tree expansion
- **Goal Threshold**: Distance threshold to consider goal reached
- **Max Iterations**: Maximum number of iterations before giving up

## Example Obstacle Definitions

```python

https://github.com/user-attachments/assets/c158b77d-04f8-4e26-8675-c876efed3cdf


# Circular obstacle
{'center': (x, y), 'radius': r}

# Rectangular obstacle
{'rect': (x, y, width, height)}
```

## Algorithm Parameters

- **step_size**: Controls how far the tree extends in each step (smaller = more precise but slower)
- **goal_threshold**: Distance from goal to consider it reached (smaller = more precise connection)
- **max_iterations**: Maximum number of tree extensions before stopping

