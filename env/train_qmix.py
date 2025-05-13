from agents.qmix_agent import QMIXTrainer
from traffic_env_multiagent import MultiAgentTrafficEnv
import os

# ==== Setup & Config ====
SUMO_CFG = "C:/Users/habdi/traffic_marl_project/sumo/osm.sumocfg"
USE_GUI = False
TOTAL_EPISODES = 100
MODEL_PATH = "qmix_traffic_model"

# ✅ Create SUMO-based Multi-Agent Env
env = MultiAgentTrafficEnv(sumo_cfg=SUMO_CFG, use_gui=USE_GUI)

# ✅ Dynamically detect obs_dim and full state_dim
example_obs = env.reset()
obs_dim = len(example_obs[env.agents[0]])
state_dim = obs_dim * len(env.agents)

# ✅ Create QMIX Trainer with correct dimensions
print(f"🔍 Detected obs_dim: {obs_dim} | state_dim: {state_dim}")
trainer = QMIXTrainer(env, obs_dim=obs_dim, state_dim=state_dim)

# ==== Train ====
print(f"🎯 Training QMIX for {TOTAL_EPISODES} episodes...")
trainer.train(episodes=TOTAL_EPISODES)

# ==== Save ====
os.makedirs(MODEL_PATH, exist_ok=True)
trainer.save(MODEL_PATH)

print(f"✅ QMIX model saved to: {MODEL_PATH}/")
