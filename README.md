# RRT Path Planning Visualization

A Python implementation of RRT path planning algorithms with real-time visualization. Supports multiple algorithm variants including standard RRT, bidirectional RRT (BiRRT), and RRT* with optimal rewiring.

https://github.com/user-attachments/assets/c8ed1382-fc40-4ed7-8092-7aeb137087ef

## Features

- **Multiple Algorithms**: RRT, Bidirectional RRT (BiRRT), and RRT* with rewiring
- **Interactive GUI**: Real-time visualization of tree growth
- **Interactive Controls**: Mode toggle, speed slider, and reset button
- **Path Planning**: Finds collision-free paths from start to goal
- **Obstacle Avoidance**: Supports circular and rectangular obstacles
- **Path Highlighting**: Automatically highlights the optimal path once the goal is reached

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

### Interactive Controls

- **Click on plot**: Set the goal position
- **Mode button**: Toggle between Unidirectional RRT, Bidirectional RRT, and RRT*
- **Speed slider**: Adjust animation speed (left = slower, right = faster)
- **Reset button**: Clear current tree and set a new goal

## Algorithms

### RRT (Unidirectional)
Standard RRT algorithm that grows a single tree from the start position toward the goal.

### BiRRT (Bidirectional)
Grows two trees simultaneously - one from the start and one from the goal. Trees merge when they connect, typically finding paths faster than unidirectional RRT.

### RRT*
An optimized version of RRT that performs rewiring to improve path quality. When adding new nodes, RRT* finds nearby nodes and rewires them if connecting through the new node reduces their path cost, resulting in shorter paths over time.

## How It Works

1. **Tree Growth**: Algorithm randomly samples points in the workspace
2. **Nearest Node**: Finds the nearest existing node to each random sample
3. **Steering**: Creates a new node by steering from nearest node toward the sample (with maximum step size)
4. **Collision Checking**: Verifies the path is collision-free before adding nodes
5. **Rewiring (RRT*)**: Optimizes paths by rewiring nearby nodes if it improves their cost
6. **Goal Connection**: Connects to goal when within threshold distance
7. **Path Extraction**: Backtracks from goal to start to extract the final path

## Customization

You can customize the following parameters in `main()`:

- **Bounds**: Workspace boundaries `(x_min, x_max, y_min, y_max)`
- **Start**: Start position `(x, y)`
- **Goal**: Set interactively by clicking on the plot
- **Obstacles**: List of obstacles (circular or rectangular)
- **Step Size**: Maximum distance for tree expansion
- **Goal Threshold**: Distance threshold to consider goal reached

### Example Obstacle Definitions

```python
# Circular obstacle
{'center': (x, y), 'radius': r}

# Rectangular obstacle
{'rect': (x, y, width, height)}
```

## Algorithm Parameters

- **step_size**: Maximum distance for tree expansion per iteration (smaller = more precise but slower)
- **goal_threshold**: Distance from goal to consider it reached (smaller = more precise connection)
- **rewire_radius** (RRT*): Radius for finding nearby nodes to rewire (default: 2 × step_size)

