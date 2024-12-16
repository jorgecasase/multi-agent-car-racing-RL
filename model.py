import os
import gym
import gym_multi_car_racing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

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
        return np.array(self._actions[act], dtype=np.float32)

def make_env():
    env = gym.make("MultiCarRacing-v0", num_agents=1, use_random_direction=True, backwards_flag=False)
    env = DiscreteActionsWrapper(env)
    return env

# Callback para detener el entrenamiento tras X episodios
class StopTrainingOnEpisodes(BaseCallback):
    def __init__(self, max_episodes: int, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        # 'infos' contiene información por cada ambiente en cada timestep
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                # El VecMonitor añade info["episode"] cuando un episodio termina
                if "episode" in info:
                    self.episode_count += 1
                    if self.verbose > 0:
                        print(f"Episodio {self.episode_count} finalizado con recompensa {info['episode']['r']}.")
                    if self.episode_count >= self.max_episodes:
                        print(f"Se alcanzó el límite de {self.max_episodes} episodios. Deteniendo el entrenamiento.")
                        return False
        return True

if __name__ == "__main__":
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Entorno de entrenamiento con VecMonitor
    train_env = DummyVecEnv([make_env])
    train_env = VecMonitor(train_env, log_dir)

    # Crear el modelo DQN
    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=10000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        device="auto"
    )

    # Entrenar, deteniéndonos después de X episodios en lugar de timesteps
    max_episodes = 10  # Número de episodios a entrenar
    callback = StopTrainingOnEpisodes(max_episodes=max_episodes, verbose=1)
    model.learn(total_timesteps=int(1e7), callback=callback)  # se usa un valor grande y el callback detiene antes
    model.save("dqn_multi_car_racing")

    # Entorno de evaluación separado
    eval_env = DummyVecEnv([make_env])

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"Recompensa media (evaluación): {mean_reward} +/- {std_reward}")

    # Cargar datos del VecMonitor
    monitor_file = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_file):
        print("No se encontró el archivo monitor.csv, puede que no se haya completado ningún episodio.")
    else:
        df = pd.read_csv(monitor_file, skiprows=1)
        episode_rewards = df["r"]
        episodes = range(len(episode_rewards))

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, episode_rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Progress (by Episodes)")
        plt.legend()
        plt.grid(True)
        plt.show()
