import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traci
from agents.rule_based import run_rule_based

# ==== Config ====
SUMO_CONFIG = "C:/Users/habdi/traffic_marl_project/sumo/osm.sumocfg"

# ==== Helper Functions for Reward Calculation ====

def is_ev(veh_id):
    return traci.vehicle.getTypeID(veh_id) == "bus_bus"

def get_total_ev_waiting_time():
    ev_ids = [veh_id for veh_id in traci.vehicle.getIDList() if is_ev(veh_id)]
    return sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in ev_ids)

def get_ev_stop_penalty():
    penalty = 0
    for veh_id in traci.vehicle.getIDList():
        if is_ev(veh_id) and traci.vehicle.getSpeed(veh_id) < 0.1:
            penalty += 1
    return penalty

def get_average_ev_speed_bonus():
    speeds = []
    for veh_id in traci.vehicle.getIDList():
        if is_ev(veh_id):
            speeds.append(traci.vehicle.getSpeed(veh_id))
    return np.mean(speeds) if speeds else 0.0

def get_green_for_ev_bonus():
    bonus = 0
    for veh_id in traci.vehicle.getIDList():
        if is_ev(veh_id):
            lane_id = traci.vehicle.getLaneID(veh_id)
            for tl in traci.trafficlight.getIDList():
                controlled_lanes = traci.trafficlight.getControlledLanes(tl)
                if lane_id in controlled_lanes:
                    phase = traci.trafficlight.getPhase(tl)
                    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0]
                    if 'G' in logic.phases[phase].state:
                        bonus += 1
    return bonus

def get_ev_anticipation_bonus():
    bonus = 0
    for veh_id in traci.vehicle.getIDList():
        if is_ev(veh_id) and traci.vehicle.getLanePosition(veh_id) > 50:
            bonus += 1
    return bonus

def get_red_hold_penalty():
    penalty = 0
    for veh_id in traci.vehicle.getIDList():
        if is_ev(veh_id) and traci.vehicle.getSpeed(veh_id) < 0.1 and traci.vehicle.getLanePosition(veh_id) > 80:
            penalty += 1
    return penalty

def compute_reward():
    ev_wait = get_total_ev_waiting_time()
    ev_stop_penalty = get_ev_stop_penalty()
    ev_speed_bonus = get_average_ev_speed_bonus()
    green_bonus = get_green_for_ev_bonus()
    anticipation_bonus = get_ev_anticipation_bonus()
    red_hold_penalty = get_red_hold_penalty()

    reward = (
        -5.0 * (ev_wait / (len(traci.vehicle.getIDList()) + 1))
        -7.0 * ev_stop_penalty
        -5.0 * red_hold_penalty
        +3.0 * ev_speed_bonus
        +4.0 * green_bonus
        +3.0 * anticipation_bonus
    )
    reward = np.clip(reward, -30, 30)
    return reward

# ==== Run Rule-Based Evaluation ====

print("🚦 Running Rule-Based Controller...")
rule_metrics = run_rule_based(SUMO_CONFIG)

# ==== Convert to DataFrame ====
rule_df = pd.DataFrame.from_dict(rule_metrics, orient='index')
rule_df['ev_id'] = rule_df.index
rule_df.reset_index(drop=True, inplace=True)

# ==== Metrics Tracking ====
rule_rewards = []
avg_queue_lengths = []

# Start SUMO and simulate manually for extra data collection
traci.start(["sumo-gui", "-c", SUMO_CONFIG, "--start"])
step = 0
MAX_STEPS = 1800

while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    reward = compute_reward()
    rule_rewards.append(reward)

    lane_ids = traci.lane.getIDList()
    total_halted = sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lane_ids)
    avg_queue_length = total_halted / len(lane_ids) if lane_ids else 0
    avg_queue_lengths.append(avg_queue_length)

    step += 1

traci.close()

# ==== Summary Stats ====
rule_avg_wait = rule_df['wait_time'].mean()
rule_avg_travel = rule_df['travel_time'].mean()

print("\n📊 Rule-Based Summary:")
print(f"Average EV Waiting Time: {rule_avg_wait:.2f} seconds")
print(f"Average EV Travel Time: {rule_avg_travel:.2f} seconds")
print(f"Average Queue Length: {np.mean(avg_queue_lengths):.2f} vehicles")
print(f"Average Reward: {np.mean(rule_rewards):.2f}")
print(f"Total EVs Observed: {len(rule_df)}")

# ==== Plotting ====
plt.figure(figsize=(10, 5))
plt.plot(rule_df['wait_time'], label='EV Wait Time (per vehicle)', marker='o', linestyle='-')
plt.title("EV Wait Time - Rule-Based Controller")
plt.xlabel("Vehicle Index")
plt.ylabel("Waiting Time (seconds)")
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(rule_rewards, label='Reward per Step (Rule-Based)', color='green')
plt.title("Reward Over Time - Rule-Based Controller")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

# ==== Export to CSV ====
rule_step_data = pd.DataFrame({
    "timestep": list(range(len(rule_rewards))),
    "reward": rule_rewards,
    "queue_length": avg_queue_lengths
})

# Combine with EV-specific data
rule_ev_data = rule_df[["ev_id", "wait_time", "travel_time"]]
rule_ev_data.to_csv("rule_ev_summary.csv", index=False)
rule_step_data.to_csv("rule_eval_timeseries.csv", index=False)

print("📁 Rule-Based results saved to rule_ev_summary.csv and rule_eval_timeseries.csv")

plt.tight_layout()
plt.show()
