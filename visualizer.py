import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.animation as animation
import pandas as pd
from tabulate import tabulate

class MazeVisualizer:
    def __init__(self, env):
        self.env = env
        self.fig = None
        self.ax = None
    
    def plot_training_progress(self, rewards, steps):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Smooth the rewards for better visualization
        window_size = min(100, len(rewards) // 10)
        if window_size > 0 and len(rewards) > window_size:
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(smoothed_rewards)
        else:
            ax1.plot(rewards)
        
        ax1.set_title('Average Reward per Episode (Smoothed)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        ax1.grid(True)
        
        ax2.plot(steps)
        ax2.set_title('Steps per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_path(self, agent, max_steps=50):
        """Get the path using the trained agent"""
        self.env.reset()
        path = [self.env.current_pos]
        
        state = self.env.get_state_index(self.env.current_pos)
        steps = 0
        
        while not self.env.done and steps < max_steps:
            action = agent.choose_action(state, training=False)
            next_pos, _, done = self.env.step(action)
            path.append(next_pos)
            state = self.env.get_state_index(next_pos)
            steps += 1
        
        return path
    
    def plot_maze_with_path(self, agent):
        """Plot the maze with the learned path"""
        path = self.get_path(agent)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a color map for the maze
        maze_colors = self._create_maze_colors()
        
        # Display the maze
        ax.imshow(maze_colors)
        
        # Draw the path
        if len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, 1], path_array[:, 0], 'g-', linewidth=3, 
                   marker='o', markersize=8, label='Robot Path')
        
        # Add grid
        self._add_grid(ax)
        
        ax.set_title('Learned Path through the Maze', fontsize=16)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Create legend
        legend_elements = self._create_legend_elements()
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.show()
        
        # Print path info
        print(f"\nFinal path length: {len(path)} steps")
        print(f"Path: {' -> '.join([f'({p[0]},{p[1]})' for p in path])}")
        
        return path
    
    def plot_q_values(self, agent):
        """Display Q-table heatmap"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        action_titles = ['Up', 'Right', 'Down', 'Left']
        
        for i, (ax, title) in enumerate(zip(axes.flat, action_titles)):
            q_values = agent.q_table[:, i].reshape(self.env.n_rows, self.env.n_cols)
            im = ax.imshow(q_values, cmap='coolwarm', aspect='auto')
            ax.set_title(f'Q-values for action: {title}')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('Q-values for Each Action', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def animate_robot_journey(self, agent, interval=500, save_gif=False, filename='robot_journey.gif'):
        """Animate the robot's journey from start to goal step by step"""
        # Get the path
        self.env.reset()
        path = [(self.env.current_pos, None)]  # (position, action_taken)
        state = self.env.get_state_index(self.env.current_pos)
        
        max_steps = 50
        steps = 0
        
        while not self.env.done and steps < max_steps:
            action = agent.choose_action(state, training=False)
            old_pos = self.env.current_pos
            next_pos, reward, done = self.env.step(action)
            path.append((next_pos, action))
            state = self.env.get_state_index(next_pos)
            steps += 1
        
        # Create figure for animation
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Robot marker
        robot_marker, = ax.plot([], [], 'ro', markersize=20, markeredgecolor='darkred', 
                               markeredgewidth=2, label='Robot')
        
        # Path line
        path_line, = ax.plot([], [], 'g-', linewidth=3, alpha=0.5, label='Path taken')
        
        # Initialize path data
        path_x = []
        path_y = []
        
        def init():
            # Create maze colors
            maze_colors = self._create_maze_colors()
            ax.imshow(maze_colors)
            
            # Add grid
            self._add_grid(ax)
            
            # Set labels
            ax.set_title('Robot Journey Animation - Step 0', fontsize=16)
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            ax.set_xlim(-0.5, self.env.n_cols - 0.5)
            ax.set_ylim(self.env.n_rows - 0.5, -0.5)
            
            # Add legend
            legend_elements = self._create_legend_elements()
            legend_elements.extend([robot_marker, path_line])
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            return robot_marker, path_line
        
        def animate(frame):
            if frame < len(path):
                pos, action = path[frame]
                row, col = pos
                
                # Update robot position
                robot_marker.set_data([col], [row])
                
                # Update path
                path_x.append(col)
                path_y.append(row)
                path_line.set_data(path_x, path_y)
                
                # Update title with step info
                action_names = ['Up', 'Right', 'Down', 'Left']
                if action is not None:
                    action_text = f" - Action: {action_names[action]}"
                else:
                    action_text = " - Start Position"
                
                # Check if robot hit mine or reached goal
                tile_type = self.env.maze[row, col]
                status = ""
                if tile_type == 2:
                    status = " - HIT MINE! 💥"
                elif tile_type == 5:
                    status = " - GOAL REACHED! 🎯"
                
                ax.set_title(f'Robot Journey Animation - Step {frame}{action_text}{status}', 
                           fontsize=16)
            
            return robot_marker, path_line
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                     frames=len(path) + 5,  # Extra frames at the end
                                     interval=interval, blit=True, repeat=True)
        
        # Save as GIF if requested
        if save_gif:
            try:
                anim.save(filename, writer='pillow', fps=2)
                print(f"Animation saved as {filename}")
            except:
                print("Could not save GIF. Make sure 'pillow' is installed: pip install pillow")
        
        plt.show()
        return anim
    
    def show_step_by_step_journey(self, agent):
        """Show the robot's journey step by step with manual control"""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        
        # Get the complete path
        self.env.reset()
        path = [(self.env.current_pos, None, self.env.total_reward)]
        state = self.env.get_state_index(self.env.current_pos)
        
        max_steps = 50
        steps = 0
        
        while not self.env.done and steps < max_steps:
            action = agent.choose_action(state, training=False)
            next_pos, reward, done = self.env.step(action)
            path.append((next_pos, action, self.env.total_reward))
            state = self.env.get_state_index(next_pos)
            steps += 1
        
        # Create interactive plot
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.15)
        
        # Current step
        current_step = [0]
        
        # Create maze background
        maze_colors = self._create_maze_colors()
        ax.imshow(maze_colors)
        self._add_grid(ax)
        
        # Initialize robot and path
        robot_marker, = ax.plot([], [], 'ro', markersize=20, markeredgecolor='darkred', 
                               markeredgewidth=2)
        path_line, = ax.plot([], [], 'g-', linewidth=3, alpha=0.5)
        path_markers, = ax.plot([], [], 'go', markersize=8)
        
        # Action arrow
        arrow = None
        
        def update_display():
            nonlocal arrow
            
            # Remove previous arrow
            if arrow:
                arrow.remove()
                arrow = None
            
            # Get current position and action
            pos, action, total_reward = path[current_step[0]]
            row, col = pos
            
            # Update robot position
            robot_marker.set_data([col], [row])
            
            # Update path
            if current_step[0] > 0:
                path_positions = path[:current_step[0] + 1]
                path_x = [p[0][1] for p in path_positions]
                path_y = [p[0][0] for p in path_positions]
                path_line.set_data(path_x, path_y)
                path_markers.set_data(path_x[:-1], path_y[:-1])  # Don't mark current position
            else:
                path_line.set_data([], [])
                path_markers.set_data([], [])
            
            # Draw action arrow
            if action is not None and current_step[0] < len(path) - 1:
                arrow_props = dict(head_width=0.2, head_length=0.1, fc='red', ec='red', 
                                 linewidth=2, alpha=0.7)
                if action == 0:  # Up
                    arrow = ax.arrow(col, row, 0, -0.4, **arrow_props)
                elif action == 1:  # Right
                    arrow = ax.arrow(col, row, 0.4, 0, **arrow_props)
                elif action == 2:  # Down
                    arrow = ax.arrow(col, row, 0, 0.4, **arrow_props)
                elif action == 3:  # Left
                    arrow = ax.arrow(col, row, -0.4, 0, **arrow_props)
            
            # Update title
            action_names = ['Up', 'Right', 'Down', 'Left']
            tile_type = self.env.maze[row, col]
            tile_name = self._get_tile_name(tile_type)
            
            title = f'Step {current_step[0]}/{len(path)-1} | Position: ({row},{col}) | Tile: {tile_name}'
            if action is not None:
                title += f' | Next Action: {action_names[action]}'
            title += f' | Total Reward: {total_reward:.1f}'
            
            # Add status
            if tile_type == 2:
                title += ' | STATUS: HIT MINE! 💥'
            elif tile_type == 5:
                title += ' | STATUS: GOAL REACHED! 🎯'
            
            ax.set_title(title, fontsize=14)
            
            # Update Q-values display
            state_idx = row * self.env.n_cols + col
            q_values = agent.q_table[state_idx]
            q_text = f'Q-values: ↑{q_values[0]:.2f} →{q_values[1]:.2f} ↓{q_values[2]:.2f} ←{q_values[3]:.2f}'
            ax.text(0.5, -0.05, q_text, transform=ax.transAxes, ha='center', 
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            fig.canvas.draw_idle()
        
        # Button callbacks
        def next_step(event):
            if current_step[0] < len(path) - 1:
                current_step[0] += 1
                update_display()
        
        def prev_step(event):
            if current_step[0] > 0:
                current_step[0] -= 1
                update_display()
        
        def reset(event):
            current_step[0] = 0
            update_display()
        
        def play_all(event):
            import time
            for i in range(current_step[0], len(path)):
                current_step[0] = i
                update_display()
                plt.pause(0.5)
        
        # Create buttons
        ax_prev = plt.axes([0.2, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.31, 0.05, 0.1, 0.04])
        ax_reset = plt.axes([0.42, 0.05, 0.1, 0.04])
        ax_play = plt.axes([0.53, 0.05, 0.1, 0.04])
        
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        btn_reset = Button(ax_reset, 'Reset')
        btn_play = Button(ax_play, 'Play All')
        
        btn_prev.on_clicked(prev_step)
        btn_next.on_clicked(next_step)
        btn_reset.on_clicked(reset)
        btn_play.on_clicked(play_all)
        
        # Set plot properties
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_xlim(-0.5, self.env.n_cols - 0.5)
        ax.set_ylim(self.env.n_rows - 0.5, -0.5)
        
        # Initial display
        update_display()
        
        plt.show()
        
        return path
    
    def _create_maze_colors(self):
        """Create color map for the maze"""
        maze_colors = np.zeros((*self.env.maze.shape, 3))
        for i in range(self.env.n_rows):
            for j in range(self.env.n_cols):
                if self.env.maze[i, j] == 1:  # Wall
                    maze_colors[i, j] = [0, 0, 0]  # Black
                elif self.env.maze[i, j] == 2:  # Mine
                    maze_colors[i, j] = [1, 0, 0]  # Red
                elif self.env.maze[i, j] == 3:  # Power
                    maze_colors[i, j] = [1, 1, 0]  # Yellow
                elif self.env.maze[i, j] == 4:  # Start
                    maze_colors[i, j] = [0, 1, 0]  # Green
                elif self.env.maze[i, j] == 5:  # Goal
                    maze_colors[i, j] = [0, 0, 1]  # Blue
                else:  # Empty
                    maze_colors[i, j] = [1, 1, 1]  # White
        return maze_colors
    
    def _add_grid(self, ax):
        """Add grid lines to the plot"""
        for i in range(self.env.n_rows + 1):
            ax.axhline(y=i - 0.5, color='gray', linewidth=0.5)
        for i in range(self.env.n_cols + 1):
            ax.axvline(x=i - 0.5, color='gray', linewidth=0.5)
    
    def _create_legend_elements(self):
        """Create legend elements"""
        return [
            Patch(facecolor='green', label='Start'),
            Patch(facecolor='blue', label='Goal'),
            Patch(facecolor='red', label='Mine'),
            Patch(facecolor='yellow', label='Power-up'),
            Patch(facecolor='black', label='Wall'),
        ]
    
    def save_qtable_to_csv(self, agent, filename='qtable_data.csv'):
        """Save Q-table to CSV file with readable format"""
        rows = []
        
        for state in range(agent.n_states):
            row = state // self.env.n_cols
            col = state % self.env.n_cols
            
            # Get tile type
            tile_type = self.env.maze[row, col]
            tile_name = self._get_tile_name(tile_type)
            
            # Get Q-values for all actions
            q_values = agent.q_table[state]
            
            # Find best action
            best_action_idx = np.argmax(q_values)
            best_action = self.env.action_names[best_action_idx]
            
            rows.append({
                'State': state,
                'Position': f'({row},{col})',
                'Tile': tile_name,
                'Q_Up': q_values[0],
                'Q_Right': q_values[1],
                'Q_Down': q_values[2],
                'Q_Left': q_values[3],
                'Best_Action': best_action,
                'Max_Q_Value': np.max(q_values)
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False, float_format='%.4f')
        print(f"Q-table saved to {filename}")
        return df
    
    def display_qtable_summary(self, agent):
        """Display a summary of the Q-table"""
        print("\n" + "="*60)
        print("Q-TABLE SUMMARY")
        print("="*60)
        
        # Get states with highest Q-values
        max_q_per_state = np.max(agent.q_table, axis=1)
        top_states_idx = np.argsort(max_q_per_state)[-1000:][::-1]
        
        print("\nTop States by Max Q-value:")
        print("-"*60)
        
        data = []
        for idx in top_states_idx:
            row = idx // self.env.n_cols
            col = idx % self.env.n_cols
            tile_type = self.env.maze[row, col]
            tile_name = self._get_tile_name(tile_type)
            
            q_values = agent.q_table[idx]
            best_action_idx = np.argmax(q_values)
            
            data.append([
                f"({row},{col})",
                tile_name,
                f"{q_values[0]:.2f}",
                f"{q_values[1]:.2f}",
                f"{q_values[2]:.2f}",
                f"{q_values[3]:.2f}",
                self.env.action_names[best_action_idx]
            ])
        
        headers = ["Position", "Tile", "Q↑", "Q→", "Q↓", "Q←", "Best"]
        print(tabulate(data, headers=headers, tablefmt="grid"))
    
    def plot_qtable_statistics(self, agent):
        """Plot various statistics about the Q-table"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Distribution of Q-values
        ax = axes[0, 0]
        all_q_values = agent.q_table.flatten()
        ax.hist(all_q_values, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title('Distribution of All Q-values')
        ax.set_xlabel('Q-value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # 2. Max Q-value per state
        ax = axes[0, 1]
        max_q_per_state = np.max(agent.q_table, axis=1)
        max_q_grid = max_q_per_state.reshape(self.env.n_rows, self.env.n_cols)
        im = ax.imshow(max_q_grid, cmap='viridis', aspect='auto')
        ax.set_title('Max Q-value per State')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax)
        
        # 3. Best action per state
        ax = axes[1, 0]
        best_actions = np.argmax(agent.q_table, axis=1)
        best_actions_grid = best_actions.reshape(self.env.n_rows, self.env.n_cols)
        
        # Create custom colormap for actions
        cmap = plt.cm.get_cmap('tab10', 4)
        im = ax.imshow(best_actions_grid, cmap=cmap, aspect='auto')
        ax.set_title('Best Action per State')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Create colorbar with action labels
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(self.env.action_names)
        
        # 4. Q-value differences (best vs second best)
        ax = axes[1, 1]
        sorted_q = np.sort(agent.q_table, axis=1)
        q_diff = sorted_q[:, -1] - sorted_q[:, -2]  # Difference between best and second best
        q_diff_grid = q_diff.reshape(self.env.n_rows, self.env.n_cols)
        im = ax.imshow(q_diff_grid, cmap='coolwarm', aspect='auto')
        ax.set_title('Q-value Margin (Best - Second Best)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle('Q-Table Statistics', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def create_policy_map(self, agent):
        """Create a visual policy map showing best actions"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create base maze visualization
        maze_colors = self._create_maze_colors()
        ax.imshow(maze_colors)
        
        # Add arrows for best actions
        arrow_props = dict(head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        for i in range(self.env.n_rows):
            for j in range(self.env.n_cols):
                if self.env.maze[i, j] not in [1, 2, 5]:  # Not wall, mine, or goal
                    state = i * self.env.n_cols + j
                    best_action = np.argmax(agent.q_table[state])
                    
                    # Arrow directions: up, right, down, left
                    if best_action == 0:  # Up
                        ax.arrow(j, i, 0, -0.3, **arrow_props)
                    elif best_action == 1:  # Right
                        ax.arrow(j, i, 0.3, 0, **arrow_props)
                    elif best_action == 2:  # Down
                        ax.arrow(j, i, 0, 0.3, **arrow_props)
                    elif best_action == 3:  # Left
                        ax.arrow(j, i, -0.3, 0, **arrow_props)
        
        # Add grid
        self._add_grid(ax)
        
        ax.set_title('Policy Map (Best Action per State)', fontsize=16)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_xlim(-0.5, self.env.n_cols - 0.5)
        ax.set_ylim(self.env.n_rows - 0.5, -0.5)
        
        # Legend
        from matplotlib.patches import FancyArrow
        legend_elements = self._create_legend_elements()
        legend_elements.append(
            FancyArrow(0, 0, 0.3, 0, head_width=0.1, head_length=0.1, 
                      fc='black', ec='black', label='Best Action')
        )
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.show()
    
    def _get_tile_name(self, tile_type):
        """Get readable name for tile type"""
        tile_names = {
            0: 'Empty',
            1: 'Wall',
            2: 'Mine',
            3: 'Power',
            4: 'Start',
            5: 'Goal'
        }
        return tile_names.get(tile_type, 'Unknown')
    
    def show_training_episode(self, agent, path, outcome, episode_num, reward):
        """Show a single training episode with outcome"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create maze colors
        maze_colors = self._create_maze_colors()
        ax.imshow(maze_colors)
        
        # Draw path
        if len(path) > 1:
            path_array = np.array(path)
            # Color based on outcome
            if outcome == 'success':
                color = 'green'
                marker = 'o'
                label = 'Successful Path'
            elif outcome == 'mine':
                color = 'red'
                marker = 'x'
                label = 'Hit Mine!'
            else:  # timeout
                color = 'orange'
                marker = 's'
                label = 'Timeout'
            
            ax.plot(path_array[:, 1], path_array[:, 0], color=color, linewidth=3, 
                   marker=marker, markersize=8, label=label, alpha=0.7)
            
            # Mark start and end
            ax.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=15, 
                   markeredgecolor='darkgreen', markeredgewidth=2, label='Start')
            ax.plot(path_array[-1, 1], path_array[-1, 0], 'ro', markersize=15, 
                   markeredgecolor='darkred', markeredgewidth=2, label='End')
        
        # Add grid
        self._add_grid(ax)
        
        # Title with outcome info
        outcome_emoji = {'success': '✅', 'mine': '💥', 'timeout': '⏱️'}
        title = f'Training Episode {episode_num} - {outcome.upper()} {outcome_emoji[outcome]}\n'
        title += f'Reward: {reward:.1f} | Steps: {len(path)-1} | Epsilon: {agent.epsilon:.3f}'
        ax.set_title(title, fontsize=16)
        
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_outcomes(self, outcomes, window_size=50):
        """Plot training outcomes over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Convert outcomes to numbers
        outcome_nums = []
        for outcome in outcomes:
            if outcome == 'success':
                outcome_nums.append(1)
            elif outcome == 'mine':
                outcome_nums.append(-1)
            else:  # timeout
                outcome_nums.append(0)
        
        episodes = range(1, len(outcomes) + 1)
        
        # Plot 1: Individual outcomes
        colors = []
        for outcome in outcomes:
            if outcome == 'success':
                colors.append('green')
            elif outcome == 'mine':
                colors.append('red')
            else:
                colors.append('orange')
        
        ax1.scatter(episodes, outcome_nums, c=colors, alpha=0.6, s=20)
        ax1.set_yticks([-1, 0, 1])
        ax1.set_yticklabels(['Mine Hit', 'Timeout', 'Success'])
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Outcome')
        ax1.set_title('Training Outcomes per Episode')
        ax1.grid(True, alpha=0.3)
        
        # Add outcome legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Success'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Timeout'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Mine Hit')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Success rate over time (moving average)
        success_rate = []
        mine_rate = []
        timeout_rate = []
        
        for i in range(len(outcomes)):
            start_idx = max(0, i - window_size + 1)
            window = outcomes[start_idx:i + 1]
            
            success_count = window.count('success')
            mine_count = window.count('mine')
            timeout_count = window.count('timeout')
            total = len(window)
            
            success_rate.append(success_count / total * 100)
            mine_rate.append(mine_count / total * 100)
            timeout_rate.append(timeout_count / total * 100)
        
        ax2.plot(episodes, success_rate, 'g-', linewidth=2, label='Success Rate')
        ax2.plot(episodes, mine_rate, 'r-', linewidth=2, label='Mine Hit Rate')
        ax2.plot(episodes, timeout_rate, 'orange', linewidth=2, label='Timeout Rate')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel(f'Rate (%) - {window_size} Episode Moving Average')
        ax2.set_title('Training Performance Over Time')
        ax2.set_ylim(-5, 105)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def animate_training_progress(self, agent, episode_paths, interval=1000, show_every=10):
        """Animate training progress showing different outcomes"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def animate(frame_idx):
            ax.clear()
            
            # Get episode data
            episode = frame_idx * show_every
            if episode >= len(episode_paths):
                episode = len(episode_paths) - 1
            
            path, outcome, reward = episode_paths[episode]
            
            # Display maze
            maze_colors = self._create_maze_colors()
            ax.imshow(maze_colors)
            
            # Draw path with color based on outcome
            if len(path) > 1:
                path_array = np.array(path)
                
                if outcome == 'success':
                    color = 'green'
                    marker = 'o'
                elif outcome == 'mine':
                    color = 'red'
                    marker = 'x'
                else:  # timeout
                    color = 'orange'
                    marker = 's'
                
                ax.plot(path_array[:, 1], path_array[:, 0], color=color, 
                       linewidth=3, marker=marker, markersize=8, alpha=0.7)
            
            # Add grid
            self._add_grid(ax)
            
            # Title
            outcome_emoji = {'success': '✅', 'mine': '💥', 'timeout': '⏱️'}
            title = f'Training Progress - Episode {episode + 1}\n'
            title += f'Outcome: {outcome.upper()} {outcome_emoji[outcome]} | '
            title += f'Reward: {reward:.1f} | Steps: {len(path)-1}'
            ax.set_title(title, fontsize=16)
            
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
        
        frames = len(episode_paths) // show_every + 1
        ani = animation.FuncAnimation(fig, animate, frames=frames, 
                                    interval=interval, repeat=True)
        plt.show()
        return ani