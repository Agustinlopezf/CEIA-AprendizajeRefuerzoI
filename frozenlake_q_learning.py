import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Crear entorno
env = gym.make("FrozenLake-v1", is_slippery=True)

# Inicializamos Q y la política
Q = defaultdict(lambda: np.zeros(env.action_space.n))
policy = defaultdict(lambda: np.ones(env.action_space.n) / env.action_space.n)

# Parámetros
n_episodes = 100000
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 0.1  # Para ε-greedy
reward_history = []

# Función para elegir acción con política ε-greedy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, env.action_space.n - 1)  # Exploración
    return np.argmax(Q[state])  # Explotación

# Bucle principal con Q-learning
for episode in tqdm(range(n_episodes), desc="Entrenando episodios"):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Actualización de Q-learning
        best_next_action = np.argmax(Q[next_state])
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][best_next_action] - Q[state][action]
        )

        # Actualizar política ε-greedy
        best_action = np.argmax(Q[state])
        policy[state] = np.ones(env.action_space.n) * epsilon / env.action_space.n
        policy[state][best_action] += (1 - epsilon)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)

# Depuración: contar éxitos
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
plt.title('Convergencia de Q-learning en FrozenLake')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Guardar el gráfico
plt.savefig('frozenlake_qlearning.png', dpi=300, bbox_inches='tight')
plt.show()