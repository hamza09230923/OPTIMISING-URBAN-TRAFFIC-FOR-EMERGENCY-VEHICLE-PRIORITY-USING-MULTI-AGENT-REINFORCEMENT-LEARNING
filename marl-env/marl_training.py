import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# Controller names
controllers = ['PPO', 'QMIX', 'Rule-Based', 'A2C']

# Evaluation metrics
avg_rewards = [27.58, 18.82, 5.41, -23.40]
ev_wait_times = [0.40, 20.27, 5.70, 607.01]
ev_travel_times = [10.37, 23.53, 473.90, 60.35]
queue_lengths = [0.00, 0.01, 0.02, 0.04]

# Create subplots for each metric
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Average Reward
axs[0, 0].bar(controllers, avg_rewards, color=['blue', 'green', 'red', 'orange'])
axs[0, 0].set_title('Average Reward per Controller')
axs[0, 0].set_ylabel('Average Reward')
axs[0, 0].grid(True, linestyle='--', alpha=0.6)

# Average EV Waiting Time
axs[0, 1].bar(controllers, ev_wait_times, color=['blue', 'green', 'red', 'orange'])
axs[0, 1].set_title('Average EV Waiting Time')
axs[0, 1].set_ylabel('Time (seconds)')
axs[0, 1].grid(True, linestyle='--', alpha=0.6)

# Average EV Travel Time
axs[1, 0].bar(controllers, ev_travel_times, color=['blue', 'green', 'red', 'orange'])
axs[1, 0].set_title('Average EV Travel Time')
axs[1, 0].set_ylabel('Time (seconds)')
axs[1, 0].grid(True, linestyle='--', alpha=0.6)

# Average Queue Length
axs[1, 1].bar(controllers, queue_lengths, color=['blue', 'green', 'red', 'orange'])
axs[1, 1].set_title('Average Queue Length')
axs[1, 1].set_ylabel('Vehicles')
axs[1, 1].grid(True, linestyle='--', alpha=0.6)

# Layout adjustments
for ax in axs.flat:
    ax.set_ylim(bottom=0)  # start y-axis at 0

plt.suptitle('Final Evaluation Metrics for All Controllers', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

