import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

class WrapperRecompensaNegativa(gym.Wrapper):
    """
    Clase wrapper para modificar las recompensas en el entorno.
    Agrega penalizaciones por pasos y por caer en agujeros.
    """
    def __init__(self, env, step_penalty=-0.005, hole_penalty=-0.3):
        super().__init__(env)
        self.step_penalty = step_penalty  # Penalización por cada paso
        self.hole_penalty = hole_penalty  # Penalización por caer en un agujero

    def step(self, action):
        """
        Sobrescribe el método step para aplicar las penalizaciones personalizadas.
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if terminated and reward == 0:  # Cayó en un agujero
            reward = self.hole_penalty
        elif not terminated and not truncated:  # Paso sin terminación
            reward = self.step_penalty
        return next_state, reward, terminated, truncated, info

# Crear entorno con el wrapper
env = gym.make("FrozenLake-v1", is_slippery=False)  # Entorno FrozenLake sin resbalones
env = WrapperRecompensaNegativa(env, step_penalty=-0.01, hole_penalty=-0.3)  # Aplicar wrapper con penalizaciones

# Inicializamos Q como un diccionario por defecto con arrays de ceros
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Parámetros del algoritmo
n_episodes = 200000  # Número de episodios de entrenamiento
alpha_start = 0.2  # Tasa de aprendizaje inicial
alpha_end = 0.01  # Tasa de aprendizaje mínima
alpha_decay = 0.9999  # Factor de decaimiento para alpha
gamma = 0.99  # Factor de descuento
epsilon_start = 1.0  # Epsilon inicial para exploración
epsilon_end = 0.01  # Epsilon mínimo
epsilon_decay = 0.99998  # Factor de decaimiento para epsilon
reward_history = []  # Historial de recompensas totales por episodio
success_history = []  # Historial de éxitos (1 si alcanzó el objetivo)
episode_lengths = []  # Longitudes de los episodios en pasos
hole_falls = []  # Historial de caídas en agujeros (1 si cayó)

# Función para elegir acción con política ε-greedy
def choose_action(state, epsilon):
    """
    Elige una acción usando ε-greedy: explora con probabilidad epsilon, explota otherwise.
    """
    if random.random() < epsilon:
        return random.randint(0, env.action_space.n - 1)  # Acción aleatoria
    return np.argmax(Q[state])  # Mejor acción conocida

# Entrenamiento con Q-learning
epsilon = epsilon_start  # Inicializar epsilon
alpha = alpha_start  # Inicializar alpha
for episode in tqdm(range(n_episodes), desc="Entrenando episodios"):
    state, _ = env.reset()  # Reiniciar entorno
    state = str(state)  # Convertir estado a string para usarlo como clave en Q
    done = False  # Bandera de terminación
    total_reward = 0  # Recompensa total del episodio
    steps = 0  # Contador de pasos
    fell_in_hole = False  # Bandera si cayó en agujero
    reached_goal = False  # Bandera si alcanzó el objetivo

    while not done:
        action = choose_action(state, epsilon)  # Elegir acción
        next_state, reward, terminated, truncated, _ = env.step(action)  # Ejecutar acción
        next_state = str(next_state)  # Convertir próximo estado a string
        done = terminated or truncated  # Verificar si terminó

        if terminated and reward == -0.3:  # Detectar caída en agujero
            fell_in_hole = True
        elif reward == 1:  # Detectar llegada al objetivo
            reached_goal = True

        best_next_action = np.argmax(Q[next_state])  # Mejor acción en el próximo estado
        # Actualizar valor Q usando la fórmula de Q-learning
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][best_next_action] - Q[state][action]
        )

        state = next_state  # Actualizar estado actual
        total_reward += reward  # Acumular recompensa
        steps += 1  # Incrementar contador de pasos

    # Decaimiento de epsilon y alpha
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    alpha = max(alpha_end, alpha * alpha_decay)
    reward_history.append(total_reward)  # Guardar recompensa total
    success_history.append(1 if reached_goal else 0)  # Guardar éxito basado en reached_goal
    episode_lengths.append(steps)  # Guardar longitud del episodio
    hole_falls.append(1 if fell_in_hole else 0)  # Guardar si cayó en agujero

    # Cada 10,000 episodios, realizar prueba de evaluación
    if episode % 10000 == 0:
        test_successes = 0
        for _ in range(100):  # 100 pruebas
            state, _ = env.reset()
            state = str(state)
            done = False
            while not done:
                action = choose_action(state, epsilon=0)  # Política greedy (sin exploración)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = str(next_state)
                done = terminated or truncated
                state = next_state
                if reward == 1:  # Éxito si recompensa es 1
                    test_successes += 1
        print(f"Episodio {episode}: Tasa de éxito en prueba = {test_successes / 100 * 100:.2f}%")

# Calcular y mostrar métricas finales
success_count = sum(success_history)  # Conteo de éxitos
hole_fall_count = sum(hole_falls)  # Conteo de caídas
print(f"Éxitos (alcanzó objetivo): {success_count}/{n_episodes}")
print(f"Tasa de éxito: {success_count / n_episodes * 100:.2f}%")
print(f"Caídas en agujeros: {hole_fall_count}/{n_episodes}")
print(f"Tasa de caídas: {hole_fall_count / n_episodes * 100:.2f}%")
print(f"Longitud promedio de episodio: {np.mean(episode_lengths):.2f} pasos")

# Función para calcular media móvil
def moving_average(data, window_size):
    """
    Calcula la media móvil de una lista de datos con una ventana dada.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

window = 1000  # Tamaño de ventana para media móvil
avg_rewards = moving_average(reward_history, window)  # Media móvil de recompensas
avg_success = moving_average(success_history, window)  # Media móvil de éxitos
avg_hole_falls = moving_average(hole_falls, window)  # Media móvil de caídas

# Generar gráficos de convergencia
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(avg_rewards, label=f'Media móvil (ventana={window})')  # Graficar media móvil
plt.plot(reward_history, alpha=0.2, label='Recompensas crudas')  # Graficar datos crudos con opacidad
plt.xlabel('Episodios')
plt.ylabel('Recompensa promedio')
plt.title('Convergencia de Recompensas')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(avg_success, label=f'Media móvil (ventana={window})')
plt.plot(success_history, alpha=0.2, label='Éxitos crudos')
plt.xlabel('Episodios')
plt.ylabel('Tasa de éxito')
plt.title('Convergencia de Tasa de Éxito')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(avg_hole_falls, label=f'Media móvil (ventana={window})')
plt.plot(hole_falls, alpha=0.2, label='Caídas crudas')
plt.xlabel('Episodios')
plt.ylabel('Tasa de caídas en agujeros')
plt.title('Convergencia de Caídas en Agujeros')
plt.grid(True)
plt.legend()

plt.tight_layout()  # Ajustar layout para evitar solapamientos
plt.savefig('frozenlake_qlearning.png', dpi=300, bbox_inches='tight')  # Guardar imagen
plt.show()  # Mostrar gráficos