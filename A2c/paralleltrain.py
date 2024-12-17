import os
import gym
import gym_multi_car_racing
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

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
        return np.array(self._actions[int(act)], dtype=np.float32)

def make_env():
    env = gym.make("MultiCarRacing-v0", num_agents=1, use_random_direction=True, backwards_flag=False)
    env = DiscreteActionsWrapper(env)
    return env

def make_multi_env(num_envs):
    return DummyVecEnv([make_env for _ in range(num_envs)])

# Callback mejorado
class StopTrainingOnEpisodes(BaseCallback):
    def __init__(self, max_episodes: int, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.global_episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.global_episode_count += 1
                if self.verbose > 0:
                    print(f"Episodio global {self.global_episode_count} finalizado con recompensa {info['episode']['r']}.")
                if self.global_episode_count >= self.max_episodes:
                    print(f"Se alcanzó el límite global de {self.max_episodes} episodios. Deteniendo el entrenamiento.")
                    return False
        return True

if __name__ == "__main__":
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    num_envs = 10  # Reducido para evitar episodios simultáneos excesivos
    train_env = make_multi_env(num_envs)
    train_env = VecMonitor(train_env, log_dir)

    model = A2C(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        vf_coef=0.25,
        ent_coef=0.01,
        verbose=1,
        device="cuda"
    )

    max_episodes = 5000
    callback = StopTrainingOnEpisodes(max_episodes=max_episodes, verbose=1)
    model.learn(total_timesteps=int(5e7), callback=callback)  # Ajustado para más entornos
    model.save("a2c_multi_car_racing")

    eval_env = DummyVecEnv([make_env])
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"Recompensa media (evaluación): {mean_reward} +/- {std_reward}")
    print("Entrenamiento finalizado.")
