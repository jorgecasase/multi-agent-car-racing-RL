import os
import time
import numpy as np
import gym
import gym_multi_car_racing
from stable_baselines3 import PPO, A2C, DQN

# Wrapper para discretizar acciones
class DiscreteActionsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._actions = [
            [-0.5, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.5], [-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]
        ]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, acts):
        # Manejar acciones para múltiples agentes
        if isinstance(acts, (list, np.ndarray)):
            return [self._actions[int(act)] for act in acts]  # Procesa cada acción
        else:
            return self._actions[int(acts)]  # Procesa una sola acción

# Función para crear el entorno con múltiples agentes
def make_env():
    env = gym.make("MultiCarRacing-v0", num_agents=3, use_random_direction=True, backwards_flag=False)
    env = DiscreteActionsWrapper(env)
    return env

if __name__ == "__main__":
    # Cargar los modelos
    model_ppo = PPO("CnnPolicy", make_env(), verbose=1)
    model_a2c = A2C("CnnPolicy", make_env(), verbose=1)
    model_dqn = DQN("CnnPolicy", make_env(), verbose=1, buffer_size=50000)

    # Inicializar entorno multiagente
    env = make_env()
    obs = env.reset()
    total_rewards = [0, 0, 0]
    done = False

    print("Iniciando simulación con PPO, A2C y DQN...")

    while not done:
        # Extraer observaciones para cada agente
        obs_ppo = obs[0]  # Observación del agente 1
        obs_a2c = obs[1]  # Observación del agente 2
        obs_dqn = obs[2]  # Observación del agente 3

        # Predecir acciones
        action_ppo, _ = model_ppo.predict(obs_ppo, deterministic=True)
        action_a2c, _ = model_a2c.predict(obs_a2c, deterministic=True)
        action_dqn, _ = model_dqn.predict(obs_dqn, deterministic=True)

        # Combinar acciones para el entorno
        actions = [action_ppo, action_a2c, action_dqn]

        # Paso en el entorno
        obs, rewards, done, _ = env.step(actions)

        # Acumular recompensas
        total_rewards[0] += rewards[0]
        total_rewards[1] += rewards[1]
        total_rewards[2] += rewards[2]

        # Renderizar
        env.render(mode="human")
        time.sleep(0.05)

    # Resultados finales
    print("\n--- Resultados finales ---")
    print(f"Agente 1 (PPO): Recompensa total = {total_rewards[0]}")
    print(f"Agente 2 (A2C): Recompensa total = {total_rewards[1]}")
    print(f"Agente 3 (DQN): Recompensa total = {total_rewards[2]}")

    env.close()
    print("Simulación completada.")
