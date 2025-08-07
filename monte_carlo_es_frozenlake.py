import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Crear entorno
env = gym.make("FrozenLake-v1", is_slippery=True)

# Inicializamos Q, returns y la política
Q = defaultdict(lambda: np.zeros(env.action_space.n))
returns = defaultdict(list)
policy = defaultdict(lambda: np.ones(env.action_space.n) / env.action_space.n)

# Parámetros
n_episodes = 1000  
gamma = 0.99
epsilon = 0.1  # Para ε-greedy
reward_history = []

for episode in tqdm(range(n_episodes), desc="Entrenando episodios"):
    # Exploring Starts: estado inicial del entorno, acción aleatoria
    state, _ = env.reset()
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

            # Política ε-greedy
            best_action = np.argmax(Q[s_t])
            policy[s_t] = np.ones(env.action_space.n) * epsilon / env.action_space.n
            policy[s_t][best_action] += (1 - epsilon)

# Depuración: contar éxitos
success_count = sum(1 for r in reward_history if r == 1)
print(f"Éxitos (recompensa = 1): {success_count}/{n_episodes}")

# Gráfico de convergencia
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window = 100  # Reducido
avg_rewards = moving_average(reward_history, window)

plt.figure(figsize=(10, 5))
plt.plot(avg_rewards, label=f'Media móvil (ventana={window})')
plt.plot(reward_history, alpha=0.3, label='Recompensas crudas')  # Añadir recompensas crudas
plt.xlabel('Episodios')
plt.ylabel('Recompensa promedio')
plt.title('Convergencia de Monte Carlo ES en FrozenLake')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Guardar el gráfico
plt.savefig('frozenlake_montecarlo_es.png', dpi=300, bbox_inches='tight')
plt.show()