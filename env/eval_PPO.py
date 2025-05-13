# run_ppo_evaluation.py
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from traffic_env import TrafficEnv
import traci

# ==== Configuration ====
SUMO_CONFIG_PATH = "C:/Users/habdi/traffic_marl_project/sumo/osm.sumocfg"
MODEL_PATH = "ppo_traffic"
USE_GUI = True
EVAL_EPISODE_LENGTH = 1800
SMOOTHING_WINDOW = 20

# ==== Load Environment ====
env = TrafficEnv(SUMO_CONFIG_PATH, use_gui=USE_GUI, max_steps=EVAL_EPISODE_LENGTH)
vec_env = DummyVecEnv([lambda: env])
model = PPO.load(MODEL_PATH, env=vec_env)
print("✅ Loaded PPO model from", MODEL_PATH)

# ==== Run Evaluation ====
obs = vec_env.reset()
done = False
step_count = 0
total_reward = 0
bus_waiting_times = []
queue_lengths = []
ev_travel_times = []
rewards = []

print("\n🚦 Starting PPO evaluation...\n")

while not done and step_count < EVAL_EPISODE_LENGTH:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    bus_waiting_times.append(env.get_total_ev_waiting_time())
    queue_lengths.append(env.get_average_queue_length())
    rewards.append(reward[0])
    ev_travel_times.append(env.get_average_ev_travel_time())

    total_reward += reward[0]
    step_count += 1

# ==== Summary Output ====
print("\n✅ PPO Evaluation Complete.")
print(f"Total steps: {step_count}")
print(f"Total reward: {total_reward:.2f}")
print(f"Average EV Waiting Time: {np.mean(bus_waiting_times):.2f} seconds")
print(f"Average EV Travel Time: {np.mean(ev_travel_times):.2f} seconds")
print(f"Average Queue Length: {np.mean(queue_lengths):.2f} vehicles")
print(f"Average Reward: {np.mean(rewards):.2f}")

# ==== Collect per-EV final stats ====
ev_data = []
ev_ids = [veh_id for veh_id in traci.vehicle.getIDList() if env._is_ev(veh_id)]

for veh_id in ev_ids:
    try:
        wait_time = traci.vehicle.getWaitingTime(veh_id)
        travel_time = traci.vehicle.getAccumulatedWaitingTime(veh_id) + traci.vehicle.getSpeed(veh_id)
        ev_data.append({"ev_id": veh_id, "wait_time": wait_time, "travel_time": travel_time})
    except traci.TraCIException:
        continue

# ==== Save to CSV ====
ppo_ev_df = pd.DataFrame(ev_data)
ppo_ev_df.to_csv("ppo_eval_results.csv", index=False)
print("📁 PPO EV results saved to ppo_eval_results.csv (ev_id, wait_time, travel_time)")

# ==== Plotting ====
timesteps = list(range(step_count))
bus_waiting_series = pd.Series(bus_waiting_times).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
queue_series = pd.Series(queue_lengths).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
reward_series = pd.Series(rewards).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

plt.figure(figsize=(10, 5))
plt.plot(timesteps, bus_waiting_series, label='Smoothed EV Waiting Time', color='blue')
plt.title("EV Waiting Time Over Time - PPO Controller")
plt.xlabel("Timestep")
plt.ylabel("Waiting Time (seconds)")
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(timesteps, queue_series, label='Smoothed Average Queue Length', color='orange')
plt.title("Queue Length Over Time - PPO Controller")
plt.xlabel("Timestep")
plt.ylabel("Vehicles per lane")
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(timesteps, reward_series, label='Smoothed Reward per Step', color='green')
plt.title("Reward Over Time - PPO Controller")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()