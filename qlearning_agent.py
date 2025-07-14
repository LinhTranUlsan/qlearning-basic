import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initialize Q-learning agent
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
    
    def choose_action(self, state, training=True):
        """
        Choose an action using epsilon-greedy policy
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation: best action based on Q-values
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using the Q-learning formula
        Q(s,a) = Q(s,a) + lr * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q
    
    def decay_epsilon(self, decay_rate=0.95):
        """Decay epsilon for exploration-exploitation balance"""
        self.epsilon = max(0.01, self.epsilon * decay_rate)
    
    def reset_epsilon(self):
        """Reset epsilon to initial value"""
        self.epsilon = self.initial_epsilon
    
    def save_q_table(self, filename='q_table.npy'):
        """Save Q-table to file"""
        np.save(filename, self.q_table)
        print(f"Q-table saved to {filename}")
    
    def load_q_table(self, filename='q_table.npy'):
        """Load Q-table from file"""
        try:
            self.q_table = np.load(filename)
            print(f"Q-table loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found. Using initialized Q-table.")
    
    def get_q_values(self, state):
        """Get Q-values for a specific state"""
        return self.q_table[state]
    
    def get_best_action(self, state):
        """Get the best action for a specific state"""
        return np.argmax(self.q_table[state])