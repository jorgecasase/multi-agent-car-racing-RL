import os
import time
import gym
import gym_multi_car_racing
import numpy as np
from stable_baselines3 import DQN
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

# Callback para renderizar el entorno
class RenderCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # Renderiza el entorno en modo "human"
        self.env.render(mode="human")
        time.sleep(0.05)  # Pausa para ralentizar el render
        return True

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

    # Crear el entorno para entrenamiento y renderizado
    train_env = make_env()  # Entorno no vectorizado para renderizar

    # Crear el modelo DQN
    model = DQN(
        policy="CnnPolicy",
        env=train_env,  # Entorno no vectorizado
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

    # Combinar callbacks: render y detener por episodios
    max_episodes = 1000
    stop_callback = StopTrainingOnEpisodes(max_episodes=max_episodes, verbose=1)
    render_callback = RenderCallback(train_env)
    callbacks = [stop_callback, render_callback]

    # Entrenar el modelo con los callbacks
    model.learn(total_timesteps=int(1e7), callback=callbacks)
    model.save("dqn_multi_car_racing")

    # Evaluación final
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=5)
    print(f"Recompensa media (evaluación): {mean_reward} +/- {std_reward}")

    print("Entrenamiento finalizado. Revisa el renderizado durante el proceso.")
