from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from traffic_env import TrafficEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Create SUMO-based env
env = TrafficEnv("C:/Users/habdi/traffic_marl_project/sumo/osm.sumocfg", use_gui=False)

# Wrap in a VecEnv for SB3 compatibility
vec_env = DummyVecEnv([lambda: env])

# Train PPO
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_tensorboard/")
model.learn(total_timesteps=500_000)  # adjust as needed

# Save model
model.save("ppo_traffic")

# Done!
print("✅ PPO model trained and saved as ppo_traffic.zip")
