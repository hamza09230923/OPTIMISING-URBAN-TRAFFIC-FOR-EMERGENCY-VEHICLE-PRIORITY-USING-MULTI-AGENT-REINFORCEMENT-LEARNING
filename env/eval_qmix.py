import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import traci
from agents.qmix_agent import QMIXTrainer
from traffic_env_multiagent import MultiAgentTrafficEnv

# ==== Config ====
SUMO_CONFIG = "C:/Users/habdi/traffic_marl_project/sumo/osm.sumocfg"
USE_GUI = False
EVAL_LENGTH = 1800
SMOOTHING_WINDOW = 20
MODEL_DIR = "C:/Users/habdi/traffic_marl_project/env/qmix_traffic_model"



# ==== Setup ====
env = MultiAgentTrafficEnv(sumo_cfg=SUMO_CONFIG, use_gui=USE_GUI)
example_obs = env.reset()
obs_dim = len(example_obs[env.agents[0]])
state_dim = obs_dim * len(env.agents)

trainer = QMIXTrainer(env, obs_dim=obs_dim, state_dim=state_dim)

print(f"🧠 Agents in env: {env.agents}")
print(f"🟢 Total traffic lights: {len(env.agents)}")

# ✅ Skip training — just load model
print("🧪 Loading trained QMIX model...")
trainer.load(MODEL_DIR)

# ==== Evaluation ====
obs = env.reset()
done = False
step_count = 0
total_reward = 0
bus_waiting_times, queue_lengths, ev_travel_times, rewards = [], [], [], []

while not done and step_count < EVAL_LENGTH:
    actions = trainer.select_actions(obs, epsilon=0.0)
    next_obs, reward_dict, done_dict, _ = env.step(actions)

    bus_waiting_times.append(env.env.get_total_ev_waiting_time())
    queue_lengths.append(env.env.get_average_queue_length())
    ev_travel_times.append(env.env.get_average_ev_travel_time())
    rewards.append(list(reward_dict.values())[0])

    total_reward += rewards[-1]
    obs = next_obs
    done = done_dict["__all__"]
    step_count += 1

# ==== Summary ====
print(f"\n📊 QMIX Evaluation Summary:")
print(f"Steps: {step_count}")
print(f"Total reward: {total_reward:.2f}")
print(f"Avg EV Wait: {np.mean(bus_waiting_times):.2f}s")
print(f"Avg EV Travel: {np.mean(ev_travel_times):.2f}s")
print(f"Avg Queue: {np.mean(queue_lengths):.2f} vehicles")
print(f"Avg Reward: {np.mean(rewards):.2f}")

# ==== Save EV Stats ====
ev_data = []
for vid in traci.vehicle.getIDList():
    if env.env._is_ev(vid):
        try:
            ev_data.append({
                "ev_id": vid,
                "wait_time": traci.vehicle.getWaitingTime(vid),
                "travel_time": traci.vehicle.getAccumulatedWaitingTime(vid) + traci.vehicle.getSpeed(vid)
            })
        except traci.TraCIException:
            continue

df = pd.DataFrame(ev_data)
df.to_csv("qmix_eval_results.csv", index=False)
print("📁 Saved per-EV data to qmix_eval_results.csv")

# ==== Plotting ====
def smooth(data):
    return pd.Series(data).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

timesteps = list(range(step_count))
plt.figure(figsize=(10, 5))
plt.plot(timesteps, smooth(bus_waiting_times), label="EV Wait Time", color="blue")
plt.title("EV Waiting Time - QMIX"); plt.xlabel("Timestep"); plt.grid(); plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(timesteps, smooth(queue_lengths), label="Queue Length", color="orange")
plt.title("Queue Length - QMIX"); plt.xlabel("Timestep"); plt.grid(); plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(timesteps, smooth(rewards), label="Reward", color="green")
plt.title("Reward - QMIX"); plt.xlabel("Timestep"); plt.grid(); plt.legend()

plt.tight_layout()
plt.show()


