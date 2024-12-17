import os
import gym
import gym_multi_car_racing
import numpy as np
from stable_baselines3 import DQN, PPO

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
    env = gym.make("MultiCarRacing-v0", num_agents=2, use_random_direction=True, backwards_flag=False)
    env = DiscreteActionsWrapper(env)
    return env

if __name__ == "__main__":
    # Cargar los modelos entrenados
    dqn_model_path = "dqn_multi_car_racing.zip"
    ppo_model_path = "ppo_multi_car_racing.zip"

    if not os.path.exists(dqn_model_path) or not os.path.exists(ppo_model_path):
        raise FileNotFoundError("Modelos no encontrados. Verifica las rutas.")

    dqn_model = DQN.load(dqn_model_path)
    ppo_model = PPO.load(ppo_model_path)

    print("Modelos cargados correctamente. Iniciando simulación...")

    # Crear el entorno
    env = make_env()
    obs = env.reset()
    done = [False] * 2
    total_rewards = [0, 0]

    while not all(np.atleast_1d(done)):
        # Predicción de acciones para cada modelo
        action_dqn, _ = dqn_model.predict(obs[0], deterministic=True)
        action_ppo, _ = ppo_model.predict(obs[1], deterministic=True)

        # Acciones procesadas para cada agente
        actions = [action_dqn, action_ppo]

        # Paso en el entorno
        obs, rewards, done, _ = env.step(actions)
        total_rewards[0] += rewards[0]
        total_rewards[1] += rewards[1]

        # Renderizado
        env.render(mode="human")

    print("\n--- Resultados Finales ---")
    print(f"DQN: Recompensa total = {total_rewards[0]}")
    print(f"PPO: Recompensa total = {total_rewards[1]}")

    env.close()
    print("Simulación finalizada.")
