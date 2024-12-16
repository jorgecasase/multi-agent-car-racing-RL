import os
import gym
import gym_multi_car_racing
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
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
        return np.array(self._actions[int(act)], dtype=np.float32)

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

    # Crear entornos vectorizados con VecMonitor
    num_envs = 4  # Número de entornos paralelos
    train_env = DummyVecEnv([make_env for _ in range(num_envs)])
    train_env = VecMonitor(train_env, log_dir)

    # Crear el modelo A2C
    model = A2C(
        policy="CnnPolicy",  # Política basada en imágenes
        env=train_env,
        learning_rate=7e-4,  # Tasa de aprendizaje recomendada para A2C
        n_steps=5,  # Número de pasos antes de actualizar (para estabilidad en A2C)
        gamma=0.99,  # Factor de descuento
        vf_coef=0.25,  # Peso de la pérdida de valor
        ent_coef=0.01,  # Peso de la entropía para exploración
        verbose=1,
        device="auto"  # Usa GPU si está disponible
    )

    # Entrenar, deteniéndonos después de X episodios
    max_episodes = 10  # Ajusta según sea necesario
    callback = StopTrainingOnEpisodes(max_episodes=max_episodes, verbose=1)
    model.learn(total_timesteps=int(1e6), callback=callback)
    model.save("a2c_multi_car_racing")

    # Entorno de evaluación separado
    eval_env = DummyVecEnv([make_env])
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"Recompensa media (evaluación): {mean_reward} +/- {std_reward}")

    # Avisar que terminó el entrenamiento
    print("Entrenamiento finalizado. Revisa los logs para más detalles.")
