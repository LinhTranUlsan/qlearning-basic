# Maze layouts
"""
        Initialize the maze environment
        Legend:
        0 = empty space
        1 = wall
        2 = mine
        3 = power-up
        4 = start position
        5 = goal
"""
MAZE_LAYOUTS = {
    'default': [
        [4, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 2, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 3],
        [0, 1, 1, 2, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 5]
    ],
    
    'simple': [
        [4, 0, 0, 0, 5],
        [0, 1, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [1, 0, 0, 0, 3],
        [0, 0, 1, 0, 0]
    ],
    
    'complex': [
        [4, 0, 1, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 0, 1, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 2, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 5]
    ]
}

# Q-Learning parameters
QLEARNING_PARAMS = {
    'learning_rate': 0.1,
    'discount_factor': 0.9,
    'initial_epsilon': 0.3,
    'epsilon_decay': 0.95,
    'min_epsilon': 0.01
}

# Training parameters
TRAINING_PARAMS = {
    'n_episodes': 2000,
    'max_steps_per_episode': 100,
    'evaluation_episodes': 10
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    'figure_size': (10, 10),
    'path_color': 'green',
    'path_width': 3,
    'marker_size': 8
}