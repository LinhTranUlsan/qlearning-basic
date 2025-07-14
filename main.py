import argparse
from maze_environment import MazeEnvironment
from qlearning_agent import QLearningAgent
from trainer import train_agent, evaluate_agent, train_agent_with_visualization, train_agent_with_live_animation
from visualizer import MazeVisualizer
from config import MAZE_LAYOUTS, QLEARNING_PARAMS, TRAINING_PARAMS

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Q-Learning Maze Navigation Robot')
    parser.add_argument('--maze', type=str, default='default', 
                       choices=['default', 'simple', 'complex'],
                       help='Choose maze layout')
    parser.add_argument('--episodes', type=int, default=TRAINING_PARAMS['n_episodes'],
                       help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=QLEARNING_PARAMS['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=QLEARNING_PARAMS['discount_factor'],
                       help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=QLEARNING_PARAMS['initial_epsilon'],
                       help='Initial epsilon for exploration')
    parser.add_argument('--save-qtable', type=str, default=None,
                       help='Filename to save Q-table')
    parser.add_argument('--load-qtable', type=str, default=None,
                       help='Filename to load Q-table from')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization')
    parser.add_argument('--analyze-qtable', action='store_true',
                       help='Show detailed Q-table analysis')
    parser.add_argument('--export-qtable-csv', type=str, default=None,
                       help='Export Q-table to CSV file')
    parser.add_argument('--show-journey', action='store_true',
                       help='Show step-by-step robot journey')
    parser.add_argument('--animate', action='store_true',
                       help='Show animated robot journey')
    parser.add_argument('--save-gif', type=str, default=None,
                       help='Save animation as GIF file')
    parser.add_argument('--watch-training', action='store_true',
                       help='Watch training episodes live')
    parser.add_argument('--training-interval', type=int, default=10,
                       help='Show training every N episodes (default: 10)')
    parser.add_argument('--live-animation', action='store_true',
                       help='Show live animated training dashboard')
    parser.add_argument('--animation-speed', type=int, default=500,
                       help='Animation update interval in milliseconds (default: 500)')
    
    args = parser.parse_args()
    
    # Initialize environment
    maze_layout = MAZE_LAYOUTS[args.maze]
    env = MazeEnvironment(maze_layout)
    print(f"Maze size: {env.n_rows}x{env.n_cols}")
    print(f"Start position: {env.start_pos}")
    print(f"Goal position: {env.goal_pos}")
    
    # Initialize agent
    n_states = env.n_rows * env.n_cols
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=env.n_actions,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=args.epsilon
    )
    
    # Load Q-table if specified
    if args.load_qtable:
        agent.load_q_table(args.load_qtable)
    
    # Train the agent
    if args.live_animation:
        # Train with live animation dashboard
        visualizer = MazeVisualizer(env)
        print(f"\nStarting live animated training dashboard...")
        rewards, steps, outcomes, episode_paths = train_agent_with_live_animation(
            env, agent, visualizer, 
            n_episodes=args.episodes,
            update_interval=args.animation_speed
        )
    elif args.watch_training:
        # Train with periodic visualization
        visualizer = MazeVisualizer(env)
        print(f"\nTraining with visualization every {args.training_interval} episodes...")
        rewards, steps, outcomes, episode_paths = train_agent_with_visualization(
            env, agent, visualizer, 
            n_episodes=args.episodes,
            visualize_every=args.training_interval
        )
    else:
        # Regular training
        print(f"\nTraining the agent for {args.episodes} episodes...")
        rewards, steps, outcomes = train_agent(env, agent, n_episodes=args.episodes)
        episode_paths = None
    
    # Save Q-table if specified
    if args.save_qtable:
        agent.save_q_table(args.save_qtable)
    
    # Evaluate the agent
    print("\nEvaluating the trained agent...")
    evaluation = evaluate_agent(env, agent, n_episodes=TRAINING_PARAMS['evaluation_episodes'])
    print(f"Success rate: {evaluation['success_rate']:.1f}%")
    print(f"Mine hit rate: {evaluation['mine_rate']:.1f}%")
    print(f"Timeout rate: {evaluation['timeout_rate']:.1f}%")
    print(f"Average reward: {evaluation['average_reward']:.2f}")
    
    # Visualization
    if not args.no_viz:
        visualizer = MazeVisualizer(env)
        
        # Plot training progress
        visualizer.plot_training_progress(rewards, steps)
        
        # Plot training outcomes if available
        if outcomes:
            visualizer.plot_training_outcomes(outcomes)
        
        # Plot the learned path
        path = visualizer.plot_maze_with_path(agent)
        
        # Plot Q-values
        visualizer.plot_q_values(agent)
    
    # Show step-by-step journey if requested
    if args.show_journey or args.animate:
        visualizer = MazeVisualizer(env) if args.no_viz else visualizer
        
        if args.show_journey:
            # Interactive step-by-step viewer
            print("\nShowing interactive step-by-step journey...")
            print("Use buttons to navigate: Previous, Next, Reset, Play All")
            visualizer.show_step_by_step_journey(agent)
        
        if args.animate:
            # Automatic animation
            print("\nShowing animated journey...")
            save_gif = args.save_gif is not None
            filename = args.save_gif if save_gif else 'robot_journey.gif'
            visualizer.animate_robot_journey(agent, interval=500, 
                                           save_gif=save_gif, filename=filename)
    
    # Q-table analysis
    if args.analyze_qtable or args.export_qtable_csv:
        visualizer = MazeVisualizer(env) if (args.no_viz and not args.show_journey) else visualizer
        
        # Export Q-table to CSV if requested
        if args.export_qtable_csv:
            df = visualizer.save_qtable_to_csv(agent, args.export_qtable_csv)
            print(f"\nQ-table exported to {args.export_qtable_csv}")
        
        # Show detailed analysis if requested
        if args.analyze_qtable:
            # Display Q-table summary
            visualizer.display_qtable_summary(agent)
            
            # Create additional visualizations
            visualizer.plot_qtable_statistics(agent)
            visualizer.create_policy_map(agent)
    
    return agent, env

if __name__ == "__main__":
    agent, env = main()