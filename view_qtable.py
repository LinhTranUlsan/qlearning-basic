import numpy as np
import matplotlib.pyplot as plt

# Load Q-table
q_table = np.load('D:/Den/Humanoid Robot/0.Simulation/QLearning/trained_model.npy')

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
action_names = ['Up', 'Right', 'Down', 'Left']
maze_shape = (8, 8)  # Adjust based on your maze

for i, (ax, action) in enumerate(zip(axes.flat, action_names)):
    # Reshape Q-values for this action
    q_values = q_table[:, i].reshape(maze_shape)
    
    # Create heatmap
    im = ax.imshow(q_values, cmap='coolwarm', aspect='auto')
    ax.set_title(f'Q-values for {action}')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)

plt.suptitle('Q-table Visualization')
plt.tight_layout()
plt.show()