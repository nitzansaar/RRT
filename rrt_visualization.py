import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import random
import math
from collections import deque

class Node:
    """Represents a node in the RRT tree"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0  # Cost from start to this node

class RRT:
    """Rapidly-exploring Random Tree implementation"""
    
    def __init__(self, start, goal, bounds, obstacles=None, step_size=10, goal_threshold=15):
        """
        Initialize RRT
        
        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            bounds: (x_min, x_max, y_min, y_max) workspace bounds
            obstacles: List of obstacles (each obstacle is a dict with 'center' and 'radius' or 'rect')
            step_size: Maximum step size for tree expansion
            goal_threshold: Distance threshold to consider goal reached
        """
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.bounds = bounds
        self.obstacles = obstacles if obstacles else []
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        
        self.nodes = [self.start]
        self.goal_reached = False
        self.goal_node = None
        
    def distance(self, node1, node2):
        """Calculate Euclidean distance between two nodes"""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def is_collision_free(self, node1, node2):
        """Check if path between two nodes is collision-free"""
        # Check multiple points along the path
        num_checks = max(10, int(self.distance(node1, node2) / 2))
        for i in range(num_checks + 1):
            t = i / num_checks
            x = node1.x + t * (node2.x - node1.x)
            y = node1.y + t * (node2.y - node1.y)
            
            # Check bounds
            if not (self.bounds[0] <= x <= self.bounds[1] and 
                    self.bounds[2] <= y <= self.bounds[3]):
                return False
            
            # Check obstacles
            for obstacle in self.obstacles:
                if 'radius' in obstacle:
                    # Circular obstacle
                    dist = math.sqrt((x - obstacle['center'][0])**2 + 
                                    (y - obstacle['center'][1])**2)
                    if dist < obstacle['radius']:
                        return False
                elif 'rect' in obstacle:
                    # Rectangular obstacle
                    rect = obstacle['rect']
                    if (rect[0] <= x <= rect[0] + rect[2] and
                        rect[1] <= y <= rect[1] + rect[3]):
                        return False
        
        return True
    
    def nearest_node(self, random_point):
        """Find the nearest node to a random point"""
        min_dist = float('inf')
        nearest = None
        
        for node in self.nodes:
            dist = math.sqrt((node.x - random_point[0])**2 + 
                           (node.y - random_point[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def steer(self, from_node, to_point):
        """Steer from a node towards a point with step_size constraint"""
        dist = math.sqrt((from_node.x - to_point[0])**2 + 
                        (from_node.y - to_point[1])**2)
        
        if dist <= self.step_size:
            return Node(to_point[0], to_point[1])
        else:
            # Create new node at step_size distance
            theta = math.atan2(to_point[1] - from_node.y, 
                             to_point[0] - from_node.x)
            new_x = from_node.x + self.step_size * math.cos(theta)
            new_y = from_node.y + self.step_size * math.sin(theta)
            return Node(new_x, new_y)
    
    def extend(self):
        """Extend the tree by one step"""
        # Random sample in workspace
        random_point = (
            random.uniform(self.bounds[0], self.bounds[1]),
            random.uniform(self.bounds[2], self.bounds[3])
        )
        
        # Find nearest node
        nearest = self.nearest_node(random_point)
        
        # Steer towards random point
        new_node = self.steer(nearest, random_point)
        
        # Check if path is collision-free
        if self.is_collision_free(nearest, new_node):
            new_node.parent = nearest
            new_node.cost = nearest.cost + self.distance(nearest, new_node)
            self.nodes.append(new_node)
            
            # Check if goal is reached
            dist_to_goal = self.distance(new_node, self.goal)
            if dist_to_goal <= self.goal_threshold and not self.goal_reached:
                # Try to connect to goal
                if self.is_collision_free(new_node, self.goal):
                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost + dist_to_goal
                    self.goal_reached = True
                    self.goal_node = new_node
                    return True
            
            return True
        
        return False
    
    def get_path(self):
        """Retrieve the path from start to goal"""
        if not self.goal_reached:
            return None
        
        path = []
        current = self.goal
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        path.reverse()
        return path


class RRTVisualizer:
    """Visualization class for RRT"""
    
    def __init__(self, start, goal, bounds, obstacles=None, step_size=10, goal_threshold=15):
        self.rrt = RRT(start, goal, bounds, obstacles, step_size, goal_threshold)
        self.bounds = bounds
        self.obstacles = obstacles if obstacles else []
        
        # Enable interactive mode for real-time updates
        plt.ion()
        
        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(bounds[0] - 10, bounds[1] + 10)
        self.ax.set_ylim(bounds[2] - 10, bounds[3] + 10)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('RRT Path Planning Visualization', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        
        # Draw obstacles
        self.draw_obstacles()
        
        # Draw start and goal
        self.ax.plot(start[0], start[1], 'go', markersize=15, label='Start', zorder=5)
        self.ax.plot(goal[0], goal[1], 'ro', markersize=15, label='Goal', zorder=5)
        
        self.ax.legend(loc='upper right')
        
        # Store line objects for incremental animation
        self.tree_lines = {}  # Dictionary: node -> line object
        self.node_points = []  # List of node point objects
        self.path_line = None
        self.iter_text = None
        
        # Track last node count for incremental updates
        self.last_node_count = 1  # Start with 1 (the start node)
        
    def draw_obstacles(self):
        """Draw obstacles on the plot"""
        for obstacle in self.obstacles:
            if 'radius' in obstacle:
                circle = Circle(obstacle['center'], obstacle['radius'], 
                              color='black', alpha=0.6)
                self.ax.add_patch(circle)
            elif 'rect' in obstacle:
                rect = obstacle['rect']
                rectangle = Rectangle((rect[0], rect[1]), rect[2], rect[3],
                                   color='black', alpha=0.6)
                self.ax.add_patch(rectangle)
    
    def update_visualization(self, max_iterations=5000, animate=True, update_frequency=1):
        """Run RRT algorithm and update visualization in real-time"""
        iterations = 0
        
        # Initialize iteration counter text
        if self.iter_text is None:
            self.iter_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                          fontsize=12, verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.show(block=False)
        
        while not self.rrt.goal_reached and iterations < max_iterations:
            # Extend the tree
            added = self.rrt.extend()
            iterations += 1
            
            # Update visualization in real-time
            if animate:
                # Incrementally add new nodes and edges
                if added and len(self.rrt.nodes) > self.last_node_count:
                    self.add_new_nodes()
                    self.last_node_count = len(self.rrt.nodes)
                
                # Update iteration counter
                self.iter_text.set_text(f'Iterations: {iterations}\nNodes: {len(self.rrt.nodes)}')
                
                # Small pause for smooth animation
                plt.pause(0.001)  # Very short pause for real-time feel
        
        # Final update - draw everything
        self.draw_tree_complete()
        
        # Draw path if goal is reached
        if self.rrt.goal_reached:
            path = self.rrt.get_path()
            if path:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                
                if self.path_line:
                    self.path_line.remove()
                
                self.path_line, = self.ax.plot(path_x, path_y, 'b-', 
                                             linewidth=3, label='Path', zorder=4)
                self.ax.legend(loc='upper right')
                
                # Update final stats
                self.iter_text.set_text(
                    f'✓ Goal reached!\nIterations: {iterations}\n'
                    f'Nodes: {len(self.rrt.nodes)}\n'
                    f'Path length: {self.rrt.goal.cost:.2f}'
                )
                
                print(f"\n✓ Goal reached in {iterations} iterations!")
                print(f"✓ Path length: {self.rrt.goal.cost:.2f}")
                print(f"✓ Path has {len(path)} waypoints")
            else:
                print("✗ Goal reached but path not found")
        else:
            self.iter_text.set_text(
                f'✗ Goal not reached\nIterations: {iterations}\nNodes: {len(self.rrt.nodes)}'
            )
            print(f"\n✗ Goal not reached after {iterations} iterations")
            print("Try increasing max_iterations or adjusting parameters")
        
        plt.draw()
        plt.ioff()  # Turn off interactive mode
        return self.rrt.goal_reached
    
    def add_new_nodes(self):
        """Incrementally add new nodes and edges to the visualization"""
        # Only process nodes that haven't been drawn yet
        for i in range(self.last_node_count, len(self.rrt.nodes)):
            node = self.rrt.nodes[i]
            
            # Draw edge to parent
            if node.parent is not None:
                line, = self.ax.plot([node.parent.x, node.x], 
                                   [node.parent.y, node.y],
                                   'gray', linewidth=0.5, alpha=0.6, zorder=1)
                self.tree_lines[node] = line
            
            # Draw node point
            point, = self.ax.plot(node.x, node.y, 'ko', markersize=2, zorder=2)
            self.node_points.append(point)
    
    def draw_tree_complete(self):
        """Draw the complete tree (used for final rendering)"""
        # Clear existing lines
        for line in self.tree_lines.values():
            line.remove()
        for point in self.node_points:
            point.remove()
        
        self.tree_lines.clear()
        self.node_points.clear()
        
        # Draw all edges in the tree
        for node in self.rrt.nodes:
            if node.parent is not None:
                line, = self.ax.plot([node.parent.x, node.x], 
                                   [node.parent.y, node.y],
                                   'gray', linewidth=0.5, alpha=0.6, zorder=1)
                self.tree_lines[node] = line
        
        # Draw all nodes
        for node in self.rrt.nodes:
            point, = self.ax.plot(node.x, node.y, 'ko', markersize=2, zorder=2)
            self.node_points.append(point)


def main():
    """Main function to run RRT visualization"""
    # Define workspace bounds (x_min, x_max, y_min, y_max)
    bounds = (0, 100, 0, 100)
    
    # Define start and goal positions
    start = (10, 10)
    goal = (90, 90)
    
    # Define obstacles (optional)
    obstacles = [
        {'center': (30, 30), 'radius': 8},
        {'center': (50, 50), 'radius': 10},
        {'center': (70, 30), 'radius': 7},
        {'rect': (40, 60, 15, 10)},  # (x, y, width, height)
        {'rect': (60, 70, 20, 8)},
    ]
    
    # Create visualizer
    visualizer = RRTVisualizer(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacles=obstacles,
        step_size=5,
        goal_threshold=10
    )
    
    # Run RRT and visualize
    print("Starting RRT path planning...")
    print("Growing tree...")
    
    success = visualizer.update_visualization(
        max_iterations=5000,
        animate=True,
        update_frequency=1  # Update every iteration for real-time visualization
    )
    
    if success:
        print("\n✓ Visualization complete!")
    else:
        print("\n✗ Visualization complete (goal not reached)")
    
    plt.show()


if __name__ == "__main__":
    main()

