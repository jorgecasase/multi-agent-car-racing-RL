import os
import time
import gym
import gym_multi_car_racing
import numpy as np
from stable_baselines3 import DQN

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

    def action(self, act):
        # Asegurar que act es un entero
        return np.array(self._actions[int(act)], dtype=np.float32)

def make_env():
    env = gym.make("MultiCarRacing-v0", num_agents=1, use_random_direction=True, backwards_flag=False)
    env = DiscreteActionsWrapper(env)
    return env

if __name__ == "__main__":
    # Cargar el modelo entrenado
    model_path = "dqn_multi_car_racing.zip"  # Cambia a la ruta de tu modelo
    if not os.path.exists(model_path):
        print(f"Modelo no encontrado en {model_path}. Entrena el modelo antes de ejecutar este script.")
        exit(1)

    model = DQN.load(model_path)
    print(f"Modelo cargado desde {model_path}")

    # Crear el entorno
    env = make_env()

    # Número de replays
    num_replays = 5

    for replay in range(num_replays):
        print(f"Inicio de la replay {replay + 1}/{num_replays}")
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Predecir acción usando el modelo
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Renderizar el entorno
            env.render(mode="human")
            time.sleep(0.05)  # Pausa para ralentizar el renderizado

        print(f"Replay {replay + 1} finalizada con recompensa total: {total_reward}")

    env.close()
    print("Todas las replays han finalizado.")
