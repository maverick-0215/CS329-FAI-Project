# Reinforcement Learning Gridworld Simulator

**CS329 - Foundations of Artificial Intelligence**  
**Course Project**

An interactive web-based reinforcement learning simulator implementing classical RL algorithms on a gridworld environment.

## Webpage

[https://maverick-0215.github.io/Learning-Grid-Game-via-RL/](https://maverick-0215.github.io/CS329-FAI-Project/)

## Overview

This project implements three reinforcement learning environments with classical RL algorithms: Q-Learning, SARSA, and Monte Carlo. The simulators run entirely in the browser using vanilla JavaScript and provide real-time visualization, performance metrics, and algorithm comparison capabilities.

## Environments

### 1. Gridworld

A navigation problem where an agent learns to reach a goal while avoiding obstacles.

**State Space:** n × n discrete grid positions (configurable from 3×3 to 10×10)

**Action Space:** Four discrete actions (Up, Down, Left, Right)

**Reward Function:**
- +10 for reaching the goal state
- -0.1 for valid movement to an empty cell
- -1 for collision with obstacle or boundary

### 2. 15-Puzzle

A sliding tile puzzle where the agent learns to arrange numbered tiles in order.

**State Space:** Permutations of tile arrangements in a 4×4 grid (15 tiles + 1 empty space)

**Action Space:** Four discrete actions (Up, Down, Left, Right) - context-dependent based on empty tile position

**Reward Function:**
- Negative reward per move (encourages shorter solutions)
- Bonus for reaching goal configuration

### 3. CartPole

A control problem where an agent balances a pole on a moving cart.

**State Space:** Continuous state space discretized into bins
- Cart Position: 3 bins
- Cart Velocity: 3 bins
- Pole Angle: 12 bins
- Pole Angular Velocity: 6 bins
- Total: 1,296 discrete states

**Action Space:** Two discrete actions (Push Left, Push Right)

**Reward Function:**
- +1 for each timestep the pole remains balanced
- Episode terminates if pole falls beyond 12° or cart moves beyond 2.4 units from center

## Algorithms Implemented

### Q-Learning

Off-policy temporal difference learning algorithm.

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**Characteristics:**
- Learns optimal action-value function independent of the policy being followed
- Uses maximum Q-value of next state
- Converges to optimal policy even with exploratory behavior

### SARSA

On-policy temporal difference learning algorithm.

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```

**Characteristics:**
- Learns the value of the policy being followed
- Uses actual next action from epsilon-greedy policy
- More conservative than Q-Learning

### Monte Carlo

Episode-based learning method using first-visit updates.

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[Gt - Q(s,a)]
```

Where Gt is the return (sum of discounted rewards) from time t until episode end.

**Characteristics:**
- Learns from complete episodes
- No bootstrapping
- Unbiased estimate of value function

## Features

- Three interactive RL environments (Gridworld, 15-Puzzle, CartPole)
- Interactive environment editors for custom configurations
- Real-time visualization of training progress
- Performance charts showing steps per episode
- Statistics dashboard with convergence metrics
- Hyperparameter controls (learning rate, discount factor, episodes)
- Side-by-side algorithm comparison
- Learned policy simulation with path animation
- Convergence detection based on path optimality and stability


### Hyperparameters

- **Learning Rate (α):** Controls step size for value updates. Range: (0, 1), typical value: 0.1
- **Discount Factor (γ):** Determines importance of future rewards. Range: [0, 1), typical value: 0.9
- **Episodes:** Number of training trajectories. Range: 100-10000, recommended: 1000-2000
- **Exploration Rate (ε):** Fixed at 0.1 for epsilon-greedy action selection

**Validation Rules:**
- Learning rate must be in range (0, 1) exclusive
- Discount factor must be in range [0, 1) 
- Both α and γ cannot exceed 0.95 simultaneously (causes instability)

### Convergence Criteria

An algorithm is considered converged when it satisfies all of the following conditions:

1. Successfully reaches the goal in last 10 episodes
2. Path length is within 3 steps of the optimal path
3. Variance is less than 5 steps² over the last 10 episodes

If convergence is not achieved, the simulator suggests increasing episodes, adjusting hyperparameters, or simplifying the environment by removing obstacles.

## Technical Implementation

- **Frontend:** HTML5, CSS3, JavaScript (ES6+)
- **Rendering:** HTML5 Canvas API for performance charts
- **Architecture:** Single-page application with no external dependencies
- **State Management:** Vanilla JavaScript objects
- **Deployment:** GitHub Pages


## Team members

- A V S Manoj (23110025)
- N Eshwar Karthikeya (23110215)
- O Akash (23110225)
- M Aniruddh Reddy (23110195)


