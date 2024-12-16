import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    log_dir = "./logs/"
    monitor_file = os.path.join(log_dir, "monitor.csv")

    if not os.path.exists(monitor_file):
        print("No se encontró el archivo monitor.csv, puede que no se hayan completado episodios o el entrenamiento no se ejecutó correctamente.")
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
