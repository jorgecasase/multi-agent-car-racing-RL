import os
import gym
import gym_multi_car_racing
import numpy as np
import time  # Importar la librería time
from stable_baselines3 import DQN, PPO, A2C


# Wrapper para discretizar las acciones
class DiscreteActionsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscreteActionsWrapper, self).__init__(env)
        self._actions = [
            [-0.5, 1.0, 0.0],  # girar izq + gas
            [0.0, 1.0, 0.0],   # recto + gas
            [0.5, 1.0, 0.0],   # girar der + gas
            [0.0, 0.0, 0.5],   # recto + freno
            [-0.5, 0.0, 0.0],  # girar izq suave sin gas
            [0.5, 0.0, 0.0],   # girar der suave sin gas
        ]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, actions):
        # Procesar acciones individuales para múltiples agentes
        return np.array([self._actions[int(a)] for a in actions], dtype=np.float32)

def make_env():
    env = gym.make("MultiCarRacing-v0", num_agents=3, use_random_direction=True, backwards_flag=False)
    env = DiscreteActionsWrapper(env)
    return env

if __name__ == "__main__":
    # Rutas de los modelos entrenados
    dqn_model_path = "dqn_multi_car_racing.zip"
    ppo_model_path = "ppo_multi_car_racing.zip"
    a2c_model_path = "a2c_multi_car_racing.zip"

    # Verificar que los modelos existen
    for model_path in [dqn_model_path, ppo_model_path, a2c_model_path]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    # Cargar los modelos entrenados
    dqn_model = DQN.load(dqn_model_path)
    ppo_model = PPO.load(ppo_model_path)
    a2c_model = A2C.load(a2c_model_path)

    print("Modelos cargados correctamente. Iniciando simulación...")

    # Crear el entorno con 3 agentes
    env = make_env()
    obs = env.reset()
    done = [False] * 3
    total_rewards = [0, 0, 0]

    while not all(np.atleast_1d(done)):
        # Predicción de acciones para cada modelo
        action_dqn, _ = dqn_model.predict(obs[0], deterministic=True)
        action_ppo, _ = ppo_model.predict(obs[1], deterministic=True)
        action_a2c, _ = a2c_model.predict(obs[2], deterministic=True)

        # Acciones procesadas para cada agente
        actions = [action_dqn, action_ppo, action_a2c]

        # Paso en el entorno
        obs, rewards, done, _ = env.step(actions)
        for i in range(3):
            total_rewards[i] += rewards[i]

        # Renderizado
        env.render(mode="human")
        time.sleep(0.05)  # Pausa para ralentizar la simulación

    print("\n--- Resultados Finales ---")
    print(f"DQN: Recompensa total = {total_rewards[0]}")
    print(f"PPO: Recompensa total = {total_rewards[1]}")
    print(f"A2C: Recompensa total = {total_rewards[2]}")

    env.close()
    print("Simulación finalizada.")