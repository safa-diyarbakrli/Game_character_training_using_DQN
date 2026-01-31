import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import imageio
from dungeon_env import DungeonDQNEnv
from dungeon_train import DQNAgent

def calculate_confidence_interval(data, confidence=0.95):
    """Calculates the 95% confidence interval for a list of data"""
    n = len(data)
    mean = np.mean(data)
    if n < 2: return mean, mean, mean # Handle single data point edge case
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin

def run_test_episodes(model_path='best_dqn_model.pth', num_episodes=1000, save_gifs=True):
    """Runs the agent through N episodes and collects performance data """
  
    print(f"\n Starting Testing: {num_episodes} Episodes using {model_path}")
    print("-" * 60)

    # Setup Environment & Agent
    #difficulty=1.0 (Hard) to test the final policy limits
    env = DungeonDQNEnv(render_mode="rgb_array", difficulty=1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, device)

    # Load Model
    try:
        agent.load(model_path)
        # Set epsilon to 0 to turn off random exploration!
        # We want to test what the agent "knows" (policy), not random moves
        agent.epsilon = 0.0 
    except FileNotFoundError:
        print("Model file not found! Train the agent first.")
        return None

    #Metrics Storage
    metrics = {
        'rewards': [],
        'steps': [],
        'successes': [],
        'loot_collected': [],
        'boss_defeated': [],
        'death_cause': [] # 'timeout', 'damage', 'success'
    }
    
    frames_buffer = [] # For GIF saving

    # Testing Loop
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        episode_frames = []

        while not done and step_count < 200:
            # Select best action (Greedy)
            action = agent.select_action(state, training=False)
            next_state, reward, done, _, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            step_count += 1

            # Capture frames for the first 5 episodes only (to save memory)
            if save_gifs and episode <= 5:
                frame = env.render()
                if frame is not None: episode_frames.append(frame)

        # Record Episode Outcome
        metrics['rewards'].append(total_reward)
        metrics['steps'].append(step_count)
        
        # Define Success: Reward > 50 implies boss killed + exit reached
        is_success = total_reward > 50
        metrics['successes'].append(1 if is_success else 0)
        
        # Record Milestones (Booleans)
        metrics['loot_collected'].append(env.has_loot)
        metrics['boss_defeated'].append(not env.boss_alive)

        # Determine Death Cause
        if is_success: cause = "Victory"
        elif step_count >= 200: cause = "Timeout"
        else: cause = "Died"
        metrics['death_cause'].append(cause)

        # Store frames for GIF generation
        if episode_frames: frames_buffer.append((episode, episode_frames, is_success))

        # Minimal Progress Print
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} | Avg Reward (Last 10): {np.mean(metrics['rewards'][-10:]):.1f}")

    env.close()
    
    # Save GIFs (Optional)
    if save_gifs and frames_buffer:
        print("\nCreating GIFs...")
        for ep_num, frames, won in frames_buffer:
            label = "WIN" if won else "LOSE"
            filename = f"test_ep{ep_num}_{label}.gif"
            try:
                imageio.mimsave(filename, frames, fps=10)
                print(f"  ✓ Saved {filename}")
            except: pass

    return metrics

def print_and_plot_results(metrics):

    rewards = metrics['rewards']
    successes = metrics['successes']
    steps = metrics['steps']
    # --- Statistical Calculations ---
    # Calculate 95% Confidence Interval for stability check
    mean_rew, ci_low, ci_high = calculate_confidence_interval(rewards)
    success_rate = np.mean(successes) * 100
    avg_steps = np.mean(steps)
    
    print("\n" + "="*40)
    print("       FINAL TEST RESULTS       ")
    print("="*40)
    print(f"Sample Size:      {len(rewards)} Episodes")
    print(f"Success Rate:     {success_rate:.1f}%")
    print(f"Average Reward:   {mean_rew:.2f} (95% CI: {ci_low:.2f} - {ci_high:.2f})")
    print(f"Average Steps:    {avg_steps:.1f}")
    print("-" * 40)
    print("Milestone Completion:")
    print(f"  - Loot Collected: {np.mean(metrics['loot_collected'])*100:.1f}%")
    print(f"  - Boss Defeated:  {np.mean(metrics['boss_defeated'])*100:.1f}%")
    print("="*40 + "\n")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Testing Analysis (N={len(rewards)})", fontsize=16)

    # Reward Distribution (Histogram)
    axes[0, 0].hist(rewards, bins=20, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(mean_rew, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_rew:.1f}')
    axes[0, 0].set_title("Reward Distribution")
    axes[0, 0].set_xlabel("Score")
    axes[0, 0].legend()

    # Reward Over Time (Line Plot)
    # Moving average to show stability over the test run
    moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
    axes[0, 1].plot(rewards, alpha=0.3, label='Raw Reward')
    axes[0, 1].plot(range(9, len(rewards)), moving_avg, color='red', label='Moving Avg (10)')
    axes[0, 1].set_title("Consistency Check")
    axes[0, 1].set_xlabel("Test Episode")
    axes[0, 1].legend()

    # Outcomes (Pie Chart)
    outcomes = metrics['death_cause']
    counts = {x: outcomes.count(x) for x in set(outcomes)}
    axes[1, 0].pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', 
                   colors=['#2ecc71', '#e74c3c', '#95a5a6'])
    axes[1, 0].set_title("Episode Outcomes")

    # Milestone Bar Chart
    milestones = ['Loot', 'Boss Kill', 'Victory']
    rates = [np.mean(metrics['loot_collected'])*100, 
             np.mean(metrics['boss_defeated'])*100, 
             success_rate]
    axes[1, 1].bar(milestones, rates, color=['gold', 'purple', 'green'], edgecolor='black')
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_title("Objective Completion Rates (%)")
    
    for i, v in enumerate(rates):
        axes[1, 1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('final_test_results.png')
    print("Plots saved to 'final_test_results.png'")
    plt.show()

if __name__ == "__main__":
    metrics = run_test_episodes()
    if metrics:
        print_and_plot_results(metrics)