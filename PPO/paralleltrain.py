import os
import gym
import gym_multi_car_racing
import numpy as np
from stable_baselines3 import PPO
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

# Crear múltiples instancias del entorno
def make_multi_env(num_envs):
    return DummyVecEnv([make_env for _ in range(num_envs)])

# Callback para detener el entrenamiento tras X episodios
class StopTrainingOnEpisodes(BaseCallback):
    def __init__(self, max_episodes: int, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
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

    # Número de entornos paralelos
    num_envs = 10 # Ajusta según los recursos de tu sistema

    # Crear múltiples entornos vectorizados con VecMonitor
    train_env = make_multi_env(num_envs)
    train_env = VecMonitor(train_env, log_dir)

    # Crear el modelo PPO
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,  # Pasos antes de actualizar (batch_size = n_steps * num_envs)
        batch_size=64,
        n_epochs=10,  # Número de épocas por actualización
        gamma=0.99,
        clip_range=0.2,  # Clipping de gradientes
        vf_coef=0.5,  # Peso de la pérdida de valor
        ent_coef=0.01,  # Peso de la entropía para fomentar exploración
        verbose=1,
        device="cuda"  # Usar GPU si está disponible
    )

    # Entrenar el modelo
    max_episodes = 1000  # Cambia según tus necesidades
    callback = StopTrainingOnEpisodes(max_episodes=max_episodes, verbose=1)
    model.learn(total_timesteps=int(1e6), callback=callback)
    model.save("ppo_multi_car_racing")

    # Entorno de evaluación separado (1 solo entorno)
    eval_env = DummyVecEnv([make_env])
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"Recompensa media (evaluación): {mean_reward} +/- {std_reward}")

    print("Entrenamiento finalizado.")
