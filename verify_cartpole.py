import math
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class CartPole:
    def __init__(self):
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5  # Half the pole's length
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # Seconds between state updates
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.reset()

    def reset(self):
        # Random start state: uniform random between -0.05 and 0.05
        self.x = random.uniform(-0.05, 0.05)
        self.x_dot = random.uniform(-0.05, 0.05)
        self.theta = random.uniform(-0.05, 0.05)
        self.theta_dot = random.uniform(-0.05, 0.05)
        self.steps = 0
        return (self.x, self.x_dot, self.theta, self.theta_dot)

    def step(self, action):
        # Action 0: Left, Action 1: Right
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)

        temp = (force + self.polemass_length * self.theta_dot**2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.mass_pole * costheta**2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        self.x += self.tau * self.x_dot
        self.x_dot += self.tau * x_acc
        self.theta += self.tau * self.theta_dot
        self.theta_dot += self.tau * theta_acc

        self.steps += 1

        done = (self.x < -self.x_threshold or self.x > self.x_threshold or
                self.theta < -self.theta_threshold_radians or self.theta > self.theta_threshold_radians)

        reward = 0 if done else 1

        return (self.x, self.x_dot, self.theta, self.theta_dot), reward, done

def discretize_state(state):
    x, x_dot, theta, theta_dot = state

    # Discretize x (Cart Position) - 3 bins
    if x < -0.8: x_bin = 0
    elif x < 0.8: x_bin = 1
    else: x_bin = 2

    # Discretize x_dot (Cart Velocity) - 3 bins
    if x_dot < -0.5: x_dot_bin = 0
    elif x_dot < 0.5: x_dot_bin = 1
    else: x_dot_bin = 2

    # Discretize Theta (Angle) - 12 bins
    if theta < -0.20: theta_bin = 0
    elif theta < -0.15: theta_bin = 1
    elif theta < -0.10: theta_bin = 2
    elif theta < -0.05: theta_bin = 3
    elif theta < -0.02: theta_bin = 4
    elif theta < 0: theta_bin = 5
    elif theta < 0.02: theta_bin = 6
    elif theta < 0.05: theta_bin = 7
    elif theta < 0.10: theta_bin = 8
    elif theta < 0.15: theta_bin = 9
    elif theta < 0.20: theta_bin = 10
    else: theta_bin = 11

    # Discretize ThetaDot (Angular Velocity) - 6 bins
    if theta_dot < -1.5: theta_dot_bin = 0
    elif theta_dot < -0.5: theta_dot_bin = 1
    elif theta_dot < 0: theta_dot_bin = 2
    elif theta_dot < 0.5: theta_dot_bin = 3
    elif theta_dot < 1.5: theta_dot_bin = 4
    else: theta_dot_bin = 5

    return (x_bin, x_dot_bin, theta_bin, theta_dot_bin)

def init_Q():
    return {}

def get_Q(Q, state_key):
    if state_key not in Q:
        Q[state_key] = [0.0, 0.0]
    return Q[state_key]

def eps_greedy(Q, state_key, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)
    
    values = get_Q(Q, state_key)
    if values[0] == values[1]:
        return random.randint(0, 1)
    return 0 if values[0] > values[1] else 1

def train_q_learning(env, alpha=0.1, gamma=0.99, episodes=1000):
    Q = init_Q()
    steps_per_episode = []
    
    min_epsilon = 0.01
    max_epsilon = 1.0
    max_steps = 500
    
    for ep in range(episodes):
        decay_ratio = min(1.0, ep / (episodes * 0.5))
        epsilon = max(min_epsilon, max_epsilon - decay_ratio * (max_epsilon - min_epsilon))
        
        state = env.reset()
        state_key = discretize_state(state)
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = eps_greedy(Q, state_key, epsilon)
            next_state, reward, done = env.step(action)
            
            next_key = discretize_state(next_state)
            next_values = get_Q(Q, next_key)
            max_next = max(next_values)
            
            current_values = get_Q(Q, state_key)
            current_values[action] += alpha * (reward + gamma * max_next - current_values[action])
            
            state = next_state
            state_key = next_key
            steps += 1
            
        steps_per_episode.append(steps)
        
    return Q, steps_per_episode

# Run a quick test
env = CartPole()
print("Running Q-Learning Test...")
Q, steps = train_q_learning(env, episodes=100)
print(f"Test Complete. Average steps last 10 episodes: {np.mean(steps[-10:])}")
