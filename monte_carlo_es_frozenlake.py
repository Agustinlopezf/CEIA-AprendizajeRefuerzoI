import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# Crear entorno
env = gym.make("FrozenLake-v1", is_slippery=True)

# Inicializamos Q, returns y la política
Q = defaultdict(lambda: np.zeros(env.action_space.n))
returns = defaultdict(list)
policy = defaultdict(lambda: np.ones(env.action_space.n) / env.action_space.n)

# Parámetros
n_episodes = 50000
gamma = 0.99
reward_history = []

for episode in range(n_episodes):
    # Exploring Starts: estado y acción inicial aleatorios
    state, _ = env.reset()
    state = random.randint(0, env.observation_space.n - 1)
    action = random.randint(0, env.action_space.n - 1)

    episode_trajectory = []
    done = False
    first_step = True
    total_reward = 0

    while not done:
        if first_step:
            s, a = state, action
            first_step = False
        else:
            a = np.random.choice(np.arange(env.action_space.n), p=policy[s])
        next_state, reward, terminated, truncated, _ = env.step(a)
        episode_trajectory.append((s, a, reward))
        total_reward += reward
        done = terminated or truncated
        s = next_state

    reward_history.append(total_reward)

    # Monte Carlo ES: Primera visita
    G = 0
    visited = set()
    for t in reversed(range(len(episode_trajectory))):
        s_t, a_t, r_t = episode_trajectory[t]
        G = gamma * G + r_t

        if (s_t, a_t) not in visited:
            visited.add((s_t, a_t))
            returns[(s_t, a_t)].append(G)
            Q[s_t][a_t] = np.mean(returns[(s_t, a_t)])

            best_action = np.argmax(Q[s_t])
            policy[s_t] = np.eye(env.action_space.n)[best_action]

# ------------------ Gráfico de convergencia ------------------

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window = 500
avg_rewards = moving_average(reward_history, window)

plt.figure(figsize=(10, 5))
plt.plot(avg_rewards)
plt.xlabel(f'Episodios (media móvil de {window})')
plt.ylabel('Recompensa promedio')
plt.title('Convergencia de Monte Carlo ES en FrozenLake')
plt.grid(True)
plt.tight_layout()
plt.show()
