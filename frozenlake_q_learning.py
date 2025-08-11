import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Crear entorno
env = gym.make("FrozenLake-v1", is_slippery=True)

# Inicializamos Q
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Parámetros
n_episodes = 100000
alpha = 0.1
gamma = 0.99
epsilon = 0.1
reward_history = []

# ε-greedy
def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, env.action_space.n - 1)  # Exploración
    return np.argmax(Q[state])  # Explotación

# Entrenamiento Q-learning
for episode in tqdm(range(n_episodes), desc="Entrenando episodios"):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Actualización Q-learning
        best_next_action = np.argmax(Q[next_state])
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][best_next_action] - Q[state][action]
        )

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)

# Métrica de éxito
success_count = sum(1 for r in reward_history if r == 1)
print(f"Éxitos (recompensa = 1): {success_count}/{n_episodes}")

# Gráfico de convergencia
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

window = 100
avg_rewards = moving_average(reward_history, window)

plt.figure(figsize=(10, 5))
plt.plot(avg_rewards, label=f'Media móvil (ventana={window})')
plt.plot(reward_history, alpha=0.3, label='Recompensas crudas')
plt.xlabel('Episodios')
plt.ylabel('Recompensa promedio')
plt.title('Convergencia de Q-learning en FrozenLake (ε-greedy explícito)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('frozenlake_qlearning_explicit.png', dpi=300, bbox_inches='tight')
plt.show()
