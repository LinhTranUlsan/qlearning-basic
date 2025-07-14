import numpy as np

class MazeEnvironment:
    def __init__(self, maze_layout):
        
        self.maze = np.array(maze_layout)
        self.n_rows, self.n_cols = self.maze.shape
        
        # Find start and goal positions
        self.start_pos = tuple(np.argwhere(self.maze == 4)[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 5)[0])
        
        # Define actions: up, right, down, left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['up', 'right', 'down', 'left']
        self.n_actions = len(self.actions)
        
        # Reset to start
        self.reset()
    
    def reset(self):
        """Reset the environment to the starting position"""
        self.current_pos = self.start_pos
        self.done = False
        self.total_reward = 0
        # Reset maze to original state (restore power-ups)
        self.maze_state = self.maze.copy()
        return self.current_pos
    
    def is_valid_position(self, pos):
        """Check if a position is valid (within bounds and not a wall)"""
        row, col = pos
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return self.maze_state[row, col] != 1  # Not a wall
        return False
    
    def step(self, action):
        """Execute an action and return the new state, reward, and done flag"""
        if self.done:
            return self.current_pos, 0, True
        
        # Calculate new position
        new_row = self.current_pos[0] + self.actions[action][0]
        new_col = self.current_pos[1] + self.actions[action][1]
        new_pos = (new_row, new_col)
        
        # Check if the new position is valid
        if self.is_valid_position(new_pos):
            self.current_pos = new_pos
            
            # Calculate reward based on the tile type
            tile_type = self.maze_state[new_pos]
            
            if tile_type == 2:  # Mine
                reward = -100
                self.done = True
            elif tile_type == 3:  # Power-up
                reward = 1 - 1  # +1 for power, -1 for step = 0
                self.maze_state[new_pos] = 0  # Remove power-up after collection
            elif tile_type == 5:  # Goal
                reward = 100 - 1  # +100 for goal, -1 for step = 99
                self.done = True
            else:  # Empty space
                reward = -1  # Step penalty
        else:
            # Invalid move (wall or out of bounds)
            reward = -1  # Step penalty
        
        self.total_reward += reward
        return self.current_pos, reward, self.done
    
    def get_state_index(self, pos):
        """Convert 2D position to 1D state index"""
        return pos[0] * self.n_cols + pos[1]