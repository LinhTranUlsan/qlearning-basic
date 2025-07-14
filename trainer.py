import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def train_agent(env, agent, n_episodes=1000, max_steps_per_episode=100, verbose=True):
    """
    Train the Q-learning agent
    """
    rewards_per_episode = []
    steps_per_episode = []
    episode_outcomes = []  # Track success/failure/timeout
    
    for episode in range(n_episodes):
        state_pos = env.reset()
        state = env.get_state_index(state_pos)
        episode_reward = 0
        steps = 0
        
        while not env.done and steps < max_steps_per_episode:
            # Choose action
            action = agent.choose_action(state)
            
            # Take action
            next_pos, reward, done = env.step(action)
            next_state = env.get_state_index(next_pos)
            
            # Update Q-value
            agent.update_q_value(state, action, reward, next_state)
            
            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Determine outcome
        if env.current_pos == env.goal_pos:
            outcome = 'success'
        elif env.done and env.current_pos != env.goal_pos:
            outcome = 'mine'
        else:
            outcome = 'timeout'
        
        rewards_per_episode.append(episode_reward)
        steps_per_episode.append(steps)
        episode_outcomes.append(outcome)
        
        # Decay epsilon for exploration-exploitation balance
        if episode % 100 == 0:
            agent.decay_epsilon()
            if verbose:
                avg_reward = np.mean(rewards_per_episode[-100:]) if len(rewards_per_episode) >= 100 else np.mean(rewards_per_episode)
                recent_outcomes = episode_outcomes[-100:] if len(episode_outcomes) >= 100 else episode_outcomes
                success_rate = recent_outcomes.count('success') / len(recent_outcomes) * 100
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Success Rate: {success_rate:.1f}%, Epsilon: {agent.epsilon:.3f}")
    
    return rewards_per_episode, steps_per_episode, episode_outcomes

def train_agent_with_live_animation(env, agent, visualizer, n_episodes=100, 
                                   update_interval=5, max_steps_per_episode=100):
    """
    Train the agent with live animated visualization
    """
    # Setup figure for live animation
    fig = plt.figure(figsize=(16, 10))
    
    # Create subplots
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
    ax_maze = fig.add_subplot(gs[0, :2])  # Main maze view
    ax_stats = fig.add_subplot(gs[0, 2])   # Statistics
    ax_reward = fig.add_subplot(gs[1, :])  # Reward graph
    ax_outcome = fig.add_subplot(gs[2, :]) # Outcome graph
    
    # Initialize data storage
    rewards_per_episode = []
    steps_per_episode = []
    episode_outcomes = []
    episode_data = []
    
    # Statistics tracking
    success_count = 0
    mine_count = 0
    timeout_count = 0
    
    def train_episode(episode):
        nonlocal success_count, mine_count, timeout_count
        
        # Run one training episode
        state_pos = env.reset()
        state = env.get_state_index(state_pos)
        episode_reward = 0
        steps = 0
        path = [state_pos]
        
        while not env.done and steps < max_steps_per_episode:
            action = agent.choose_action(state)
            next_pos, reward, done = env.step(action)
            next_state = env.get_state_index(next_pos)
            path.append(next_pos)
            
            agent.update_q_value(state, action, reward, next_state)
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Determine outcome
        if env.current_pos == env.goal_pos:
            outcome = 'success'
            success_count += 1
        elif env.done and env.current_pos != env.goal_pos:
            outcome = 'mine'
            mine_count += 1
        else:
            outcome = 'timeout'
            timeout_count += 1
        
        rewards_per_episode.append(episode_reward)
        steps_per_episode.append(steps)
        episode_outcomes.append(outcome)
        episode_data.append((path, outcome, episode_reward))
        
        # Decay epsilon
        if episode % 100 == 0:
            agent.decay_epsilon()
        
        return path, outcome, episode_reward, steps
    
    def init():
        ax_maze.clear()
        ax_stats.clear()
        ax_reward.clear()
        ax_outcome.clear()
        return []
    
    def animate(episode):
        # Train for one episode
        path, outcome, reward, steps = train_episode(episode)
        
        # Clear axes
        ax_maze.clear()
        ax_stats.clear()
        
        # 1. Draw maze with current episode path
        maze_colors = visualizer._create_maze_colors()
        ax_maze.imshow(maze_colors)
        
        if len(path) > 1:
            path_array = np.array(path)
            
            # Color based on outcome
            if outcome == 'success':
                color = 'green'
                marker = 'o'
                emoji = '✅'
            elif outcome == 'mine':
                color = 'red'
                marker = 'x'
                emoji = '💥'
            else:
                color = 'orange'
                marker = 's'
                emoji = '⏱️'
            
            ax_maze.plot(path_array[:, 1], path_array[:, 0], color=color, 
                        linewidth=3, marker=marker, markersize=8, alpha=0.7)
            
            # Mark start and end
            ax_maze.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=15)
            ax_maze.plot(path_array[-1, 1], path_array[-1, 0], 'ro', markersize=15)
        
        visualizer._add_grid(ax_maze)
        ax_maze.set_title(f'Episode {episode + 1}/{n_episodes} - {outcome.upper()} {emoji}', 
                         fontsize=14)
        ax_maze.set_xlabel('Column')
        ax_maze.set_ylabel('Row')
        
        # 2. Update statistics panel
        ax_stats.text(0.1, 0.9, f'Episode: {episode + 1}', transform=ax_stats.transAxes, 
                     fontsize=12, weight='bold')
        ax_stats.text(0.1, 0.8, f'Outcome: {outcome.upper()}', transform=ax_stats.transAxes, 
                     fontsize=12)
        ax_stats.text(0.1, 0.7, f'Reward: {reward:.1f}', transform=ax_stats.transAxes, 
                     fontsize=12)
        ax_stats.text(0.1, 0.6, f'Steps: {steps}', transform=ax_stats.transAxes, 
                     fontsize=12)
        ax_stats.text(0.1, 0.5, f'Epsilon: {agent.epsilon:.3f}', transform=ax_stats.transAxes, 
                     fontsize=12)
        
        # Overall statistics
        total_episodes = episode + 1
        ax_stats.text(0.1, 0.35, 'Overall Stats:', transform=ax_stats.transAxes, 
                     fontsize=12, weight='bold')
        ax_stats.text(0.1, 0.25, f'Success: {success_count} ({success_count/total_episodes*100:.1f}%)', 
                     transform=ax_stats.transAxes, fontsize=11, color='green')
        ax_stats.text(0.1, 0.15, f'Mine Hit: {mine_count} ({mine_count/total_episodes*100:.1f}%)', 
                     transform=ax_stats.transAxes, fontsize=11, color='red')
        ax_stats.text(0.1, 0.05, f'Timeout: {timeout_count} ({timeout_count/total_episodes*100:.1f}%)', 
                     transform=ax_stats.transAxes, fontsize=11, color='orange')
        
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis('off')
        
        # 3. Update reward graph
        ax_reward.clear()
        if len(rewards_per_episode) > 1:
            episodes_range = range(1, len(rewards_per_episode) + 1)
            ax_reward.plot(episodes_range, rewards_per_episode, 'b-', alpha=0.3, linewidth=1)
            
            # Moving average
            if len(rewards_per_episode) > 10:
                window_size = min(20, len(rewards_per_episode) // 5)
                moving_avg = np.convolve(rewards_per_episode, 
                                       np.ones(window_size)/window_size, mode='valid')
                ax_reward.plot(range(window_size, len(rewards_per_episode) + 1), 
                             moving_avg, 'b-', linewidth=2, label='Moving Avg')
            
            ax_reward.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax_reward.set_xlabel('Episode')
            ax_reward.set_ylabel('Reward')
            ax_reward.set_title('Episode Rewards', fontsize=12)
            ax_reward.grid(True, alpha=0.3)
        
        # 4. Update outcome graph
        ax_outcome.clear()
        if len(episode_outcomes) > 0:
            # Count outcomes in windows
            window = min(20, len(episode_outcomes))
            success_rate = []
            mine_rate = []
            
            for i in range(len(episode_outcomes)):
                start_idx = max(0, i - window + 1)
                window_outcomes = episode_outcomes[start_idx:i + 1]
                
                s_rate = window_outcomes.count('success') / len(window_outcomes) * 100
                m_rate = window_outcomes.count('mine') / len(window_outcomes) * 100
                
                success_rate.append(s_rate)
                mine_rate.append(m_rate)
            
            episodes_range = range(1, len(episode_outcomes) + 1)
            ax_outcome.plot(episodes_range, success_rate, 'g-', linewidth=2, label='Success')
            ax_outcome.plot(episodes_range, mine_rate, 'r-', linewidth=2, label='Mine Hit')
            
            ax_outcome.set_xlabel('Episode')
            ax_outcome.set_ylabel('Rate (%)')
            ax_outcome.set_title(f'Outcome Rates ({window}-Episode Window)', fontsize=12)
            ax_outcome.set_ylim(-5, 105)
            ax_outcome.legend(loc='right')
            ax_outcome.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return []
    
    # Create animation
    print(f"\nStarting live training animation for {n_episodes} episodes...")
    print("Close the window to stop training early.\n")
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_episodes, 
                        interval=update_interval, repeat=False, blit=False)
    
    plt.show()
    
    # Print final statistics
    print(f"\nTraining completed!")
    print(f"Total episodes: {len(episode_outcomes)}")
    print(f"Final success rate: {success_count/len(episode_outcomes)*100:.1f}%")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    
    return rewards_per_episode, steps_per_episode, episode_outcomes, episode_data

def train_agent_with_visualization(env, agent, visualizer, n_episodes=100, 
                                 visualize_every=10, max_steps_per_episode=100):
    """
    Train the agent with periodic visualization of performance
    """
    rewards_per_episode = []
    steps_per_episode = []
    episode_outcomes = []
    episode_paths = []
    
    for episode in range(n_episodes):
        state_pos = env.reset()
        state = env.get_state_index(state_pos)
        episode_reward = 0
        steps = 0
        path = [state_pos]
        
        while not env.done and steps < max_steps_per_episode:
            # Choose action
            action = agent.choose_action(state)
            
            # Take action
            next_pos, reward, done = env.step(action)
            next_state = env.get_state_index(next_pos)
            path.append(next_pos)
            
            # Update Q-value
            agent.update_q_value(state, action, reward, next_state)
            
            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Determine outcome
        if env.current_pos == env.goal_pos:
            outcome = 'success'
        elif env.done and env.current_pos != env.goal_pos:
            outcome = 'mine'
        else:
            outcome = 'timeout'
        
        rewards_per_episode.append(episode_reward)
        steps_per_episode.append(steps)
        episode_outcomes.append(outcome)
        episode_paths.append((path, outcome, episode_reward))
        
        # Visualize progress
        if (episode + 1) % visualize_every == 0:
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"Last outcome: {outcome}, Reward: {episode_reward:.1f}, Steps: {steps}")
            recent_outcomes = episode_outcomes[-visualize_every:]
            success_count = recent_outcomes.count('success')
            mine_count = recent_outcomes.count('mine')
            timeout_count = recent_outcomes.count('timeout')
            print(f"Last {visualize_every} episodes: {success_count} successes, "
                  f"{mine_count} mine hits, {timeout_count} timeouts")
            
            # Show the path
            visualizer.show_training_episode(agent, path, outcome, episode + 1, episode_reward)
        
        # Decay epsilon
        if episode % 100 == 0:
            agent.decay_epsilon()
    
    return rewards_per_episode, steps_per_episode, episode_outcomes, episode_paths

def evaluate_agent(env, agent, n_episodes=10):
    """
    Evaluate the trained agent
    """
    total_rewards = []
    successful_episodes = 0
    mine_hits = 0
    timeouts = 0
    
    for _ in range(n_episodes):
        state_pos = env.reset()
        state = env.get_state_index(state_pos)
        episode_reward = 0
        steps = 0
        
        while not env.done and steps < 100:
            action = agent.choose_action(state, training=False)
            next_pos, reward, done = env.step(action)
            state = env.get_state_index(next_pos)
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        if env.current_pos == env.goal_pos:
            successful_episodes += 1
        elif env.done and env.current_pos != env.goal_pos:
            mine_hits += 1
        else:
            timeouts += 1
    
    success_rate = successful_episodes / n_episodes * 100
    mine_rate = mine_hits / n_episodes * 100
    timeout_rate = timeouts / n_episodes * 100
    avg_reward = np.mean(total_rewards)
    
    return {
        'success_rate': success_rate,
        'mine_rate': mine_rate,
        'timeout_rate': timeout_rate,
        'average_reward': avg_reward,
        'total_rewards': total_rewards,
        'successful_episodes': successful_episodes,
        'mine_hits': mine_hits,
        'timeouts': timeouts
    }