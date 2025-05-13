import os
import time
import subprocess
import webbrowser
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from traffic_env import TrafficEnv

# ==== Config ====
SUMO_CONFIG_PATH = "C:/Users/habdi/traffic_marl_project/sumo/osm.sumocfg"
USE_GUI = False
TRAIN_TIMESTEPS = 500000  # 🔼 Increased
MODEL_PATH = "a2c_traffic"
TENSORBOARD_LOG_DIR = "./tensorboard_logs/a2c_traffic"
TENSORBOARD_PORT = 6006

# ==== Launch TensorBoard ====
print("🚀 Launching TensorBoard...")
tb_process = subprocess.Popen(
    ["tensorboard", f"--logdir={TENSORBOARD_LOG_DIR}", f"--port={TENSORBOARD_PORT}"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
time.sleep(3)
webbrowser.open(f"http://localhost:{TENSORBOARD_PORT}")

# ==== Load SUMO Environment ====
env = TrafficEnv(SUMO_CONFIG_PATH, use_gui=USE_GUI, max_steps=1800)
vec_env = DummyVecEnv([lambda: env])

# ✅ Normalize observations and rewards
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# ==== A2C Hyperparameters ====
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # 🔼 Deeper policy & value net
)

model = A2C(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=3e-4,
    verbose=1,
    tensorboard_log=TENSORBOARD_LOG_DIR,
    policy_kwargs=policy_kwargs,
)

# ==== Train ====
print(f"🎯 Training A2C for {TRAIN_TIMESTEPS:,} timesteps...")
model.learn(total_timesteps=TRAIN_TIMESTEPS)

# ==== Save Model ====
model.save(MODEL_PATH)
vec_env.save(os.path.join(MODEL_PATH + "_vec_normalize.pkl"))

print(f"✅ A2C model and VecNormalize saved to: {MODEL_PATH}/")



