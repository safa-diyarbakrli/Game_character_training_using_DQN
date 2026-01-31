import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from dungeon_env import DungeonDQNEnv
import pickle

# Named tuple for storing information (experience)
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


# =============================================================================
# DEEP Q-NETWORK ARCHITECTURE
# =============================================================================

class DQN(nn.Module):
    """ Architecture:
        Input: 30-dimensional state vector
        Hidden 1: 256 units + ReLU + Dropout(0.1)
        Hidden 2: 256 units + ReLU + Dropout(0.1)
        Hidden 3: 256 units + ReLU
        Output: 5 Q-values (one per action) """
        
    def __init__(self, input_size, hidden_size=256, num_actions=5):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_actions)

        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class PrioritizedReplayBuffer:
    """
    This focuses learning on experiences the agent can learn most from.
    """
    def __init__(self, capacity=50000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use
        self.beta = beta  # Importance sampling correction
        self.beta_increment = 0.001
        # Storage for experiences and priorities
        self.buffer = []
        self.priorities = np.array([])
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer with maximum priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        # If buffer not full, append new experience
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, done))
            self.priorities = np.append(self.priorities, max_priority)
        else: # If buffer full, overwrite oldest experience
            self.buffer[self.position] = Experience(state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
        # Update position for next experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch of experiences according to priorities"""
        
        
        # Use only filled portion of buffer
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        # Calculate sampling probabilities from priorities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices according to probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences at sampled indices
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Gradually increase beta toward 1.0 
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Unpack experiences into separate lists for batch processing
        states = [e.state for e in experiences]
        actions = [e.action for e in experiences]
        rewards = [e.reward for e in experiences]
        next_states = [e.next_state for e in experiences]
        dones = [e.done for e in experiences]
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # Small epsilon to prevent zero priority
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with Prioritized Experience Replay"""
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
       # Policy network: Used for action selection and training
        self.policy_net = DQN(state_size, hidden_size=256, num_actions=action_size).to(device)
        
        # Target network: Used for computing target Q-values
        self.target_net = DQN(state_size, hidden_size=256, num_actions=action_size).to(device)
    
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 
        
        # Optimizer with lower learning rate for stability
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        
        # PRIORITIZED REPLAY BUFFER
        self.memory = PrioritizedReplayBuffer(capacity=50000, alpha=0.6, beta=0.4)
        
        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9985  
        self.target_update_frequency = 10
        
        self.steps_done = 0
        
    def select_action(self, state, training=True):
        """ Select an action using epsilon-greedy strategy """
        # EXPLORATION
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else: # EXPLOITATION
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self):
        """Perform one training step on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample prioritized batch
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate TD errors for priority update
        td_errors = (current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors)
        
        # Weighted loss
        loss = (weights * nn.SmoothL1Loss(reduction='none')(current_q_values.squeeze(), target_q_values)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network by copying weights from policy network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")



# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_dqn(num_episodes=10000, max_steps=200, save_interval=1000):
    """ Training with Curriculum Learning and Prioritized Replay """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Start easy, get harder
    curriculum_stages = [
        (0, 2000, 0.3),      # Episodes 0-2000: Easy (30% difficulty)
        (2000, 5000, 0.6),   # Episodes 2000-5000: Medium (60% difficulty)
        (5000, 10000, 1.0),  # Episodes 5000-10000: Full (100% difficulty)
    ]
    
    def get_difficulty(episode):
        for start, end, diff in curriculum_stages:
            if start <= episode < end:
                return diff
        return 1.0
    
    # Initialize with medium difficulty
    env = DungeonDQNEnv(difficulty=0.6)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, device)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    success_count = 0
    
    best_reward = -float('inf')
    
    print("=" * 70)
    print(f"\nState size: {state_size}, Action size: {action_size}")
    print(f"Total episodes: {num_episodes}")
    print("-" * 70)
    
    for episode in range(1, num_episodes + 1):
        # Adjust difficulty
        current_difficulty = get_difficulty(episode)
        # Only re-create env if difficulty changed 
        if abs(env.difficulty - current_difficulty) > 0.01:
            env.close() # Close the old one
            env = DungeonDQNEnv(difficulty=current_difficulty)
        state, _ = env.reset()
       
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, done, _, _ = env.step(action)
            
            # Store in prioritized replay
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        if episode_reward > 50:  # Success threshold
            success_count += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network
        if episode % agent.target_update_frequency == 0:
            agent.update_target_network()
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save('best_dqn_model.pth')
        
        # Periodic save
        if episode % save_interval == 0:
            agent.save(f'dqn_model_episode_{episode}.pth')
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            success_rate = success_count / 100 if episode >= 100 else success_count / episode
            
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Difficulty: {current_difficulty:.1%}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Length (last 100): {avg_length:.2f}")
            print(f"  Success Rate (last 100): {success_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Best Reward: {best_reward:.2f}")
            print(f"  Memory Size: {len(agent.memory)}")
            print(f"  Beta (PER): {agent.memory.beta:.3f}")
            print("-" * 70)
            
            success_count = 0
            
    # Plot results
    plot_training_results(episode_rewards, episode_lengths, losses)
    def save_training_data(rewards, lengths, losses, filename='training_data.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({
                'rewards': rewards,
                'lengths': lengths,
                'losses': losses
            }, f)
        print(f"Training data saved to {filename}")
        
    # Save final model
    agent.save('final_dqn_model.pth')
    save_training_data(episode_rewards, episode_lengths, losses)
    return agent, episode_rewards


def plot_training_results(rewards, lengths, losses):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards
    axes[0, 0].plot(rewards, alpha=0.3, label='Episode Reward')
    axes[0, 0].plot(moving_average(rewards, 100), label='Moving Avg (100)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(lengths, alpha=0.3, label='Episode Length')
    axes[0, 1].plot(moving_average(lengths, 100), label='Moving Avg (100)', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Losses
    if losses:
        axes[1, 0].plot(losses, alpha=0.5)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].grid(True)
    
    # Success rate
    window = 100
    success_threshold = 50
    successes = [1 if r > success_threshold else 0 for r in rewards]
    success_rate = [np.mean(successes[max(0, i-window):i+1]) for i in range(len(successes))]
    axes[1, 1].plot(success_rate, linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title(f'Success Rate (Moving Avg {window})')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("Training plots saved to 'training_results.png'")
    plt.show()


def moving_average(data, window):
    """Calculate moving average"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


if __name__ == "__main__":
 
    # Train
    agent, rewards = train_dqn(
        num_episodes=10000,
        max_steps=200,
        save_interval=1000
    )
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Final average reward (last 100): {np.mean(rewards[-100:]):.2f}")
    print("=" * 70 + "\n")
