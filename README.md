# Autonomous Dungeon Navigation using Deep Q-Networks (DQN)

## Project Overview

This project implements an autonomous agent capable of solving a complex, multi-objective "Dungeon Crawler" environment using **Deep Reinforcement Learning (DRL)**.

Unlike standard tabular Q-Learning, this agent uses a **Deep Q-Network (DQN)** to handle a continuous state space and dynamic, stochastic adversaries. The agent must navigate a 7x7 grid, collect essential items, and defeat bosses while dodging random projectile attacks ("Bullet Hell" mechanics).

## Key Features

- **Deep Q-Network (DQN):** Replaces the Q-Table with a Neural Network (3 layers, 256 neurons) to approximate Q-values for complex states.
- **Prioritized Experience Replay (PER):** The agent learns more frequently from "surprising" events (like unexpected damage or victory) rather than repetitive walking.
- **Curriculum Learning:** Training is split into 3 difficulty stages. The agent learns basic navigation first, then combat, and finally high-speed survival.
- **Stochastic Adversaries:** Bosses do not follow fixed paths. They move stochastically (20-30% frequency) and fire projectiles, forcing the agent to learn reactive policies rather than memorizing a route.

## System Architecture

### 1. The Environment (`dungeon_env.py`)

A custom Gymnasium-based grid world with:

- **State Space (30 inputs):** Normalized coordinates of Player, Bosses, and up to 5 Fireballs + Inventory flags.
- **Action Space (5 actions):** `North`, `South`, `East`, `West`, `Attack`.
- **Rewards:** Shaped rewards for sub-goals (Loot: +25, Boss: +50) and penalties for inefficiency (-0.05/step) or damage.

### 2. The Agent (`dungeon_train.py`)

- **Network:** PyTorch-based neural network.
- **Target Network:** A "frozen" copy of the policy net, updated every 10 episodes to stabilize the Bellman updates.
- **Optimization:** Adam Optimizer with Smooth L1 Loss.

## Performance Results

The model was evaluated on a rigorous **1,000-episode test set** with exploration disabled ($\epsilon=0$).

| Metric              | Result    | Description                                     |
| :------------------ | :-------- | :---------------------------------------------- |
| **Success Rate**    | **98.4%** | Cleared the dungeon in 984/1000 runs.           |
| **Loot Collection** | **99.9%** | Near-perfect navigation to the first objective. |
| **Survival Rate**   | **98.5%** | Only 1.5% of runs ended in death by fireballs.  |
| **Avg Steps**       | **34.9**  | Highly efficient pathing (down from 200+).      |
