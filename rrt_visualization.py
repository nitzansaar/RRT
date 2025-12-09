import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Button, Slider
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


class RRTStar(RRT):
    """RRT* implementation with optimal rewiring"""
    
    def __init__(self, start, goal, bounds, obstacles=None, step_size=10, goal_threshold=15, rewire_radius=None):
        """
        Initialize RRT*
        
        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            bounds: (x_min, x_max, y_min, y_max) workspace bounds
            obstacles: List of obstacles
            step_size: Maximum step size for tree expansion
            goal_threshold: Distance threshold to consider goal reached
            rewire_radius: Radius for finding nearby nodes (if None, uses adaptive radius)
        """
        super().__init__(start, goal, bounds, obstacles, step_size, goal_threshold)
        self.rewire_radius = rewire_radius if rewire_radius else step_size * 2.0  # Default radius
    
    def get_nearby_nodes(self, point, radius):
        """Find all nodes within radius of a point"""
        nearby = []
        px, py = point[0], point[1]
        for node in self.nodes:
            dist = math.sqrt((node.x - px)**2 + (node.y - py)**2)
            if dist <= radius:
                nearby.append(node)
        return nearby
    
    def choose_best_parent(self, new_node, nearby_nodes):
        """Choose the best parent from nearby nodes that minimizes cost"""
        best_parent = None
        best_cost = float('inf')
        
        for node in nearby_nodes:
            if self.is_collision_free(node, new_node):
                cost = node.cost + self.distance(node, new_node)
                if cost < best_cost:
                    best_cost = cost
                    best_parent = node
        
        return best_parent, best_cost
    
    def rewire(self, new_node, nearby_nodes):
        """Rewire nearby nodes if connecting through new_node improves their cost"""
        for node in nearby_nodes:
            if node == new_node.parent:  # Skip the parent
                continue
            
            # Check if connecting node through new_node would improve cost
            new_cost = new_node.cost + self.distance(new_node, node)
            if new_cost < node.cost:
                # Check if path is collision-free
                if self.is_collision_free(new_node, node):
                    # Rewire: change parent
                    node.parent = new_node
                    # Update cost and propagate cost change to children
                    old_cost = node.cost
                    node.cost = new_cost
                    cost_diff = new_cost - old_cost
                    self.update_children_cost(node, cost_diff)
    
    def update_children_cost(self, node, cost_diff):
        """Recursively update cost of all children after rewiring"""
        for child in self.nodes:
            if child.parent == node:
                child.cost += cost_diff
                self.update_children_cost(child, cost_diff)
    
    def extend(self):
        """Extend the tree by one step with RRT* rewiring"""
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
            # Find nearby nodes within rewire radius
            nearby_nodes = self.get_nearby_nodes((new_node.x, new_node.y), self.rewire_radius)
            
            # Choose best parent from nearby nodes
            best_parent, best_cost = self.choose_best_parent(new_node, nearby_nodes)
            
            if best_parent is not None:
                new_node.parent = best_parent
                new_node.cost = best_cost
                self.nodes.append(new_node)
                
                # Rewire nearby nodes
                self.rewire(new_node, nearby_nodes)
                
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


class BiRRT:
    """Bidirectional Rapidly-exploring Random Tree implementation"""
    
    def __init__(self, start, goal, bounds, obstacles=None, step_size=10, goal_threshold=15):
        """
        Initialize BiRRT with two trees
        
        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            bounds: (x_min, x_max, y_min, y_max) workspace bounds
            obstacles: List of obstacles (each obstacle is a dict with 'center' and 'radius' or 'rect')
            step_size: Maximum step size for tree expansion
            goal_threshold: Distance threshold to consider trees connected
        """
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.bounds = bounds
        self.obstacles = obstacles if obstacles else []
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        
        # Two trees: one from start, one from goal
        self.start_tree = [self.start]
        self.goal_tree = [self.goal]
        
        self.trees_connected = False
        self.connection_node_start = None  # Node in start tree that connects
        self.connection_node_goal = None   # Node in goal tree that connects
        
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
    
    def nearest_node(self, random_point, tree):
        """Find the nearest node to a random point in a given tree"""
        min_dist = float('inf')
        nearest = None
        
        for node in tree:
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
    
    def extend_tree(self, tree, random_point):
        """Extend a tree towards a random point"""
        # Find nearest node
        nearest = self.nearest_node(random_point, tree)
        
        # Steer towards random point
        new_node = self.steer(nearest, random_point)
        
        # Check if path is collision-free
        if self.is_collision_free(nearest, new_node):
            new_node.parent = nearest
            new_node.cost = nearest.cost + self.distance(nearest, new_node)
            tree.append(new_node)
            return new_node
        
        return None
    
    def try_connect_trees(self, new_node, from_tree, to_tree):
        """Try to connect a new node from one tree to the nearest node in another tree"""
        nearest_in_other = self.nearest_node((new_node.x, new_node.y), to_tree)
        dist = self.distance(new_node, nearest_in_other)
        
        if dist <= self.goal_threshold:
            # Try to connect directly
            if self.is_collision_free(new_node, nearest_in_other):
                self.trees_connected = True
                if from_tree == self.start_tree:
                    self.connection_node_start = new_node
                    self.connection_node_goal = nearest_in_other
                else:
                    self.connection_node_start = nearest_in_other
                    self.connection_node_goal = new_node
                return True
        
        return False
    
    def extend(self):
        """Extend both trees by one step each (alternating)"""
        # Random sample in workspace
        random_point = (
            random.uniform(self.bounds[0], self.bounds[1]),
            random.uniform(self.bounds[2], self.bounds[3])
        )
        
        # Alternate between extending start tree and goal tree
        # Extend start tree
        new_node_start = self.extend_tree(self.start_tree, random_point)
        if new_node_start:
            # Check if we can connect to goal tree
            if self.try_connect_trees(new_node_start, self.start_tree, self.goal_tree):
                return True
        
        # Try extending goal tree towards the new node from start tree (if it was created)
        if new_node_start:
            new_node_goal = self.extend_tree(self.goal_tree, (new_node_start.x, new_node_start.y))
            if new_node_goal:
                if self.try_connect_trees(new_node_goal, self.goal_tree, self.start_tree):
                    return True
        
        # Also try extending goal tree with a new random point
        random_point2 = (
            random.uniform(self.bounds[0], self.bounds[1]),
            random.uniform(self.bounds[2], self.bounds[3])
        )
        new_node_goal = self.extend_tree(self.goal_tree, random_point2)
        if new_node_goal:
            if self.try_connect_trees(new_node_goal, self.goal_tree, self.start_tree):
                return True
        
        return new_node_start is not None or new_node_goal is not None
    
    def get_path(self):
        """Retrieve the path from start to goal by merging both trees"""
        if not self.trees_connected:
            return None
        
        # Build path from start to connection point
        path_start = []
        current = self.connection_node_start
        while current is not None:
            path_start.append((current.x, current.y))
            current = current.parent
        
        path_start.reverse()
        
        # Build path from connection point to goal
        path_goal = []
        current = self.connection_node_goal
        while current is not None:
            path_goal.append((current.x, current.y))
            current = current.parent
        
        # Merge paths (path_start ends at connection, path_goal starts at connection)
        # Remove duplicate connection point
        if path_goal:
            path_goal = path_goal[1:]  # Remove first point (connection point)
        
        full_path = path_start + path_goal
        return full_path
    
    @property
    def goal_reached(self):
        """Property to check if goal is reached (trees are connected)"""
        return self.trees_connected
    
    @property
    def nodes(self):
        """Property to get all nodes from both trees"""
        return self.start_tree + self.goal_tree


class RRTVisualizer:
    """Visualization class for RRT"""
    
    def __init__(self, start, goal, bounds, obstacles=None, step_size=10, goal_threshold=15):
        self.start_pos = start
        self.goal_pos = goal
        self.bounds = bounds
        self.obstacles = obstacles if obstacles else []
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        
        # Algorithm mode: 'unidirectional', 'bidirectional', or 'rrt_star'
        self.algorithm_mode = 'unidirectional'
        
        # Will be initialized after goal selection
        self.rrt = None
        
        # Enable interactive mode for real-time updates
        plt.ion()
        
        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(bounds[0] - 10, bounds[1] + 10)
        self.ax.set_ylim(bounds[2] - 10, bounds[3] + 10)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('RRT Path Planning Visualization - Click to set goal', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        
        # Draw obstacles
        self.draw_obstacles()
        
        # Draw start
        self.start_point, = self.ax.plot(start[0], start[1], 'go', markersize=15, label='Start', zorder=5)
        
        # Goal will be drawn after selection
        self.goal_point = None
        
        self.ax.legend(loc='upper right')
        
        # Store line objects for incremental animation
        self.tree_lines = {}  # Dictionary: node -> line object
        self.node_points = []  # List of node point objects
        self.path_line = None
        self.iter_text = None
        
        # Track last node count for incremental updates
        self.last_node_count = 1  # Start with 1 (the start node)
        
        # For interactive goal selection
        self.goal_selected = False
        self.click_handler = None
        
        # Reset button
        self.reset_button = None
        self.toggle_button = None
        self.speed_slider = None
        self.running = False
        
        # Animation speed (pause duration in seconds)
        # Lower value = faster animation, higher value = slower animation
        self.animation_delay = 0.001  # Default: fast
        
        # Track nodes for bidirectional visualization
        self.last_start_tree_count = 1
        self.last_goal_tree_count = 1
        
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
    
    def reset(self):
        """Reset the visualization to allow selecting a new goal and running again"""
        # Clear the tree
        self.clear_tree()
        
        # Reset RRT
        self.rrt = None
        self.goal_selected = False
        self.last_node_count = 1
        self.last_start_tree_count = 1
        self.last_goal_tree_count = 1
        
        # Remove goal point
        if self.goal_point:
            self.goal_point.remove()
            self.goal_point = None
        
        # Clear path
        if self.path_line:
            self.path_line.remove()
            self.path_line = None
        
        # Clear iteration text
        if self.iter_text:
            self.iter_text.remove()
            self.iter_text = None
        
        # Reset title
        mode_display = {'unidirectional': 'RRT', 'bidirectional': 'BiRRT', 'rrt_star': 'RRT*'}
        mode_text = mode_display.get(self.algorithm_mode, 'RRT')
        self.ax.set_title(f'{mode_text} Path Planning Visualization - Click to set goal', fontsize=16, fontweight='bold')
        
        # Reset goal selection flag
        self.goal_selected = False
        
        # Redraw
        self.fig.canvas.draw()
        
        print("\n" + "="*50)
        print("Reset complete! Click on the plot to set a new goal.")
        print("="*50)
        
        # Reconnect click handler for new goal selection
        self.setup_goal_click_handler()
    
    def clear_tree(self):
        """Clear all tree visualization elements"""
        # Clear tree lines
        for line in self.tree_lines.values():
            line.remove()
        self.tree_lines.clear()
        
        # Clear node points
        for point in self.node_points:
            point.remove()
        self.node_points.clear()
    
    def setup_reset_button(self):
        """Create and setup the reset button"""
        # Create button axes (position: left, bottom, width, height in figure coordinates)
        ax_reset = plt.axes([0.02, 0.02, 0.1, 0.04])
        self.reset_button = Button(ax_reset, 'Reset', color='lightcoral', hovercolor='red')
        
        def reset_callback(event):
            if not self.running:
                self.reset()
                # Reconnect click handler for new goal selection
                self.setup_goal_click_handler()
        
        self.reset_button.on_clicked(reset_callback)
    
    def setup_toggle_button(self):
        """Create and setup the algorithm mode toggle button"""
        # Create button axes (position: left, bottom, width, height in figure coordinates)
        ax_toggle = plt.axes([0.13, 0.02, 0.20, 0.04])
        mode_display = {'unidirectional': 'Unidirectional', 'bidirectional': 'Bidirectional', 'rrt_star': 'RRT*'}
        self.toggle_button = Button(ax_toggle, f'Mode: {mode_display[self.algorithm_mode]}', color='lightblue', hovercolor='lightcyan')
        
        def toggle_callback(event):
            if not self.running:
                # Cycle through modes: unidirectional -> bidirectional -> rrt_star -> unidirectional
                mode_order = ['unidirectional', 'bidirectional', 'rrt_star']
                current_index = mode_order.index(self.algorithm_mode)
                next_index = (current_index + 1) % len(mode_order)
                self.algorithm_mode = mode_order[next_index]
                
                mode_text = mode_display[self.algorithm_mode]
                self.toggle_button.label.set_text(f'Mode: {mode_text}')
                print(f"\nMode switched to: {mode_text}")
                if self.rrt is not None:
                    print("Note: Mode change will take effect on next reset.")
        
        self.toggle_button.on_clicked(toggle_callback)
    
    def setup_speed_slider(self):
        """Create and setup the speed control slider"""
        # Create slider axes (position: left, bottom, width, height in figure coordinates)
        # Positioned further right to avoid overlap with mode toggle button
        ax_slider = plt.axes([0.36, 0.02, 0.25, 0.03])
        
        # Slider range: 0 (slow) to 1 (fast)
        # We'll map this inversely to pause duration: 0 = 0.1s pause, 1 = 0.0001s pause
        self.speed_slider = Slider(
            ax_slider,
            'Speed',
            0.0,  # min (slow)
            1.0,  # max (fast)
            valinit=0.9,  # Default: fast (maps to ~0.001s pause)
            valstep=0.01,
            color='lightgreen'
        )
        
        def update_speed(val):
            # Map slider value (0-1) to pause duration
            # 0 (slow) -> 0.1 seconds, 1 (fast) -> 0.0001 seconds
            # Using exponential mapping for smoother control
            slider_val = self.speed_slider.val
            # Inverse mapping: low slider value = high delay (slow), high slider value = low delay (fast)
            self.animation_delay = 0.1 * (1.0 - slider_val) ** 2 + 0.0001
        
        self.speed_slider.on_changed(update_speed)
        # Initialize the delay based on default slider value
        update_speed(0.9)
    
    def setup_goal_click_handler(self):
        """Setup click handler for goal selection"""
        # Disconnect any existing click handler
        if self.click_handler is not None:
            self.fig.canvas.mpl_disconnect(self.click_handler)
        
        def on_click(event):
            if event.inaxes != self.ax:
                return
            
            # Check if click is within bounds
            x, y = event.xdata, event.ydata
            if not (self.bounds[0] <= x <= self.bounds[1] and 
                    self.bounds[2] <= y <= self.bounds[3]):
                print(f"Click is outside bounds. Please click within the workspace.")
                return
            
            # Set goal position
            self.goal_pos = (x, y)
            
            # Remove old goal if exists
            if self.goal_point:
                self.goal_point.remove()
            
            # Draw new goal
            self.goal_point, = self.ax.plot(x, y, 'ro', markersize=15, label='Goal', zorder=5)
            self.ax.legend(loc='upper right')
            
            # Update title
            mode_display = {'unidirectional': 'RRT', 'bidirectional': 'BiRRT', 'rrt_star': 'RRT*'}
            mode_text = mode_display.get(self.algorithm_mode, 'RRT')
            self.ax.set_title(f'{mode_text} Path Planning Visualization', fontsize=16, fontweight='bold')
            
            # Initialize algorithm based on mode
            if self.algorithm_mode == 'bidirectional':
                self.rrt = BiRRT(self.start_pos, self.goal_pos, self.bounds, 
                                self.obstacles, self.step_size, self.goal_threshold)
                self.last_start_tree_count = 1
                self.last_goal_tree_count = 1
            elif self.algorithm_mode == 'rrt_star':
                self.rrt = RRTStar(self.start_pos, self.goal_pos, self.bounds, 
                                   self.obstacles, self.step_size, self.goal_threshold)
                self.last_node_count = 1
            else:  # unidirectional
                self.rrt = RRT(self.start_pos, self.goal_pos, self.bounds, 
                              self.obstacles, self.step_size, self.goal_threshold)
                self.last_node_count = 1
            
            self.goal_selected = True
            self.fig.canvas.draw()
            
            print(f"Goal set at ({x:.2f}, {y:.2f})")
            mode_display = {'unidirectional': 'RRT', 'bidirectional': 'BiRRT', 'rrt_star': 'RRT*'}
            algo_name = mode_display.get(self.algorithm_mode, 'RRT')
            print(f"Starting {algo_name} algorithm...")
            
            # Disconnect click handler
            self.fig.canvas.mpl_disconnect(self.click_handler)
            self.click_handler = None
            
            # Run algorithm
            self.run_algorithm()
        
        # Connect click handler
        self.click_handler = self.fig.canvas.mpl_connect('button_press_event', on_click)
    
    def wait_for_goal_selection(self):
        """Wait for user to click on the plot to set the goal"""
        print("Click on the plot to set the goal position...")
        self.goal_selected = False
        self.setup_goal_click_handler()
        
        # Wait for goal selection (non-blocking with pause)
        plt.show(block=False)
        while not self.goal_selected:
            plt.pause(0.1)
    
    def run_algorithm(self):
        """Run the RRT algorithm"""
        self.running = True
        success = self.update_visualization(
            max_iterations=30000,
            animate=True,
            update_frequency=1
        )
        self.running = False
        
        if success:
            print("\n✓ Visualization complete!")
        else:
            print("\n✗ Visualization complete (goal not reached)")
    
    def update_visualization(self, max_iterations=5000, animate=True, update_frequency=1):
        """Run RRT algorithm and update visualization in real-time"""
        if self.rrt is None:
            raise ValueError("RRT not initialized. Please select a goal first using wait_for_goal_selection().")
        
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
                if self.algorithm_mode == 'bidirectional' and isinstance(self.rrt, BiRRT):
                    # Handle bidirectional mode
                    if added:
                        if len(self.rrt.start_tree) > self.last_start_tree_count:
                            self.add_new_nodes_bidirectional('start')
                            self.last_start_tree_count = len(self.rrt.start_tree)
                        if len(self.rrt.goal_tree) > self.last_goal_tree_count:
                            self.add_new_nodes_bidirectional('goal')
                            self.last_goal_tree_count = len(self.rrt.goal_tree)
                    
                    # Update iteration counter
                    total_nodes = len(self.rrt.start_tree) + len(self.rrt.goal_tree)
                    self.iter_text.set_text(
                        f'Iterations: {iterations}\n'
                        f'Start Tree: {len(self.rrt.start_tree)}\n'
                        f'Goal Tree: {len(self.rrt.goal_tree)}\n'
                        f'Total Nodes: {total_nodes}'
                    )
                else:
                    # Handle unidirectional mode (RRT or RRT*)
                    if added and len(self.rrt.nodes) > self.last_node_count:
                        self.add_new_nodes()
                        self.last_node_count = len(self.rrt.nodes)
                    
                    # Update iteration counter
                    self.iter_text.set_text(f'Iterations: {iterations}\nNodes: {len(self.rrt.nodes)}')
                
                # Small pause for smooth animation (controlled by speed slider)
                plt.pause(self.animation_delay)
        
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
                if self.algorithm_mode == 'bidirectional' and isinstance(self.rrt, BiRRT):
                    total_nodes = len(self.rrt.start_tree) + len(self.rrt.goal_tree)
                    path_length = sum(
                        math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
                        for i in range(len(path) - 1)
                    )
                    self.iter_text.set_text(
                        f'✓ Goal reached!\nIterations: {iterations}\n'
                        f'Start Tree: {len(self.rrt.start_tree)}\n'
                        f'Goal Tree: {len(self.rrt.goal_tree)}\n'
                        f'Total Nodes: {total_nodes}\n'
                        f'Path length: {path_length:.2f}'
                    )
                    print(f"\n✓ Goal reached in {iterations} iterations!")
                    print(f"✓ Path length: {path_length:.2f}")
                    print(f"✓ Path has {len(path)} waypoints")
                else:
                    # RRT or RRT*
                    path_length = self.rrt.goal.cost if hasattr(self.rrt.goal, 'cost') else sum(
                        math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
                        for i in range(len(path) - 1)
                    )
                    self.iter_text.set_text(
                        f'✓ Goal reached!\nIterations: {iterations}\n'
                        f'Nodes: {len(self.rrt.nodes)}\n'
                        f'Path length: {path_length:.2f}'
                    )
                    print(f"\n✓ Goal reached in {iterations} iterations!")
                    print(f"✓ Path length: {path_length:.2f}")
                    print(f"✓ Path has {len(path)} waypoints")
            else:
                print("✗ Goal reached but path not found")
        else:
            if self.algorithm_mode == 'bidirectional' and isinstance(self.rrt, BiRRT):
                total_nodes = len(self.rrt.start_tree) + len(self.rrt.goal_tree)
                self.iter_text.set_text(
                    f'✗ Goal not reached\nIterations: {iterations}\n'
                    f'Start Tree: {len(self.rrt.start_tree)}\n'
                    f'Goal Tree: {len(self.rrt.goal_tree)}\n'
                    f'Total Nodes: {total_nodes}'
                )
            else:
                self.iter_text.set_text(
                    f'✗ Goal not reached\nIterations: {iterations}\nNodes: {len(self.rrt.nodes)}'
                )
            print(f"\n✗ Goal not reached after {iterations} iterations")
            print("Try increasing max_iterations or adjusting parameters")
        
        plt.draw()
        # Keep interactive mode on for reset button
        return self.rrt.goal_reached
    
    def add_new_nodes(self):
        """Incrementally add new nodes and edges to the visualization (unidirectional)"""
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
    
    def add_new_nodes_bidirectional(self, tree_type):
        """Incrementally add new nodes and edges for bidirectional visualization"""
        if not isinstance(self.rrt, BiRRT):
            return
        
        tree = self.rrt.start_tree if tree_type == 'start' else self.rrt.goal_tree
        last_count = self.last_start_tree_count if tree_type == 'start' else self.last_goal_tree_count
        
        # Color coding: start tree = blue/green tint, goal tree = red/orange tint
        if tree_type == 'start':
            edge_color = 'steelblue'
            node_color = 'darkblue'
        else:
            edge_color = 'coral'
            node_color = 'darkred'
        
        # Only process nodes that haven't been drawn yet
        for i in range(last_count, len(tree)):
            node = tree[i]
            
            # Draw edge to parent
            if node.parent is not None:
                line, = self.ax.plot([node.parent.x, node.x], 
                                   [node.parent.y, node.y],
                                   edge_color, linewidth=0.5, alpha=0.6, zorder=1)
                self.tree_lines[node] = line
            
            # Draw node point
            point, = self.ax.plot(node.x, node.y, node_color, marker='o', markersize=2, zorder=2)
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
        
        if self.algorithm_mode == 'bidirectional' and isinstance(self.rrt, BiRRT):
            # Draw start tree (blue)
            for node in self.rrt.start_tree:
                if node.parent is not None:
                    line, = self.ax.plot([node.parent.x, node.x], 
                                       [node.parent.y, node.y],
                                       'steelblue', linewidth=0.5, alpha=0.6, zorder=1)
                    self.tree_lines[node] = line
                point, = self.ax.plot(node.x, node.y, 'darkblue', marker='o', markersize=2, zorder=2)
                self.node_points.append(point)
            
            # Draw goal tree (red/orange)
            for node in self.rrt.goal_tree:
                if node.parent is not None:
                    line, = self.ax.plot([node.parent.x, node.x], 
                                       [node.parent.y, node.y],
                                       'coral', linewidth=0.5, alpha=0.6, zorder=1)
                    self.tree_lines[node] = line
                point, = self.ax.plot(node.x, node.y, 'darkred', marker='o', markersize=2, zorder=2)
                self.node_points.append(point)
        else:
            # Draw all edges in the tree (unidirectional)
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
    
    # Define start position
    start = (10, 10)
    # Default goal (will be replaced by user click)
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
        goal=goal,  # Default goal, will be replaced by user click
        bounds=bounds,
        obstacles=obstacles,
        step_size=5,
        goal_threshold=10
    )
    
    # Setup reset button, toggle button, and speed slider
    visualizer.setup_reset_button()
    visualizer.setup_toggle_button()
    visualizer.setup_speed_slider()
    
    # Wait for user to click and set the goal
    visualizer.wait_for_goal_selection()
    
    # Run RRT and visualize
    print("Growing tree...")
    visualizer.run_algorithm()
    
    # Keep window open for reset button
    print("\nYou can click the 'Reset' button to run again with a new goal.")
    plt.show(block=True)  # Keep window open


if __name__ == "__main__":
    main()

