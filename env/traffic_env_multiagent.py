import gym
import numpy as np
from traffic_env import TrafficEnv

class MultiAgentTrafficEnv:
    def __init__(self, sumo_cfg="sumo/osm.sumocfg", use_gui=False, max_steps=1800):
        self.env = TrafficEnv(sumo_cfg, use_gui, max_steps)
        self.agents = self.env.traffic_lights
        self.n_agents = len(self.agents)

        # Each agent gets part of the total obs (split vector)
        obs_example, _ = self.env.reset()
        obs_split = np.array_split(obs_example, self.n_agents)
        self.obs_dim = len(obs_split[0])

        self.observation_space = {
            agent: gym.spaces.Box(low=0, high=100, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_space = {
            agent: gym.spaces.Discrete(4) for agent in self.agents
        }

    def reset(self):
        obs, _ = self.env.reset()
        split_obs = np.array_split(obs, self.n_agents)
        return {agent: split_obs[i] for i, agent in enumerate(self.agents)}

    def step(self, actions):
        action_list = [actions[agent] for agent in self.agents]
        obs, reward, done, _, info = self.env.step(action_list)

        split_obs = np.array_split(obs, self.n_agents)
        obs_dict = {agent: split_obs[i] for i, agent in enumerate(self.agents)}
        reward_dict = {agent: reward for agent in self.agents}
        done_dict = {agent: done for agent in self.agents}
        done_dict["__all__"] = done

        return obs_dict, reward_dict, done_dict, info

    def close(self):
        self.env.close()


