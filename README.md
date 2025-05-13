# Optimising Urban Traffic for Emergency Vehicle Priority Using Multi-Agent Reinforcement Learning 🚦🚑

This project explores the application of **Multi-Agent Reinforcement Learning (MARL)** to optimise traffic signal control in urban environments, with a special focus on **prioritising emergency vehicles (EVs)** such as ambulances. By leveraging decentralized and cooperative agents, we aim to reduce EV waiting and travel times while maintaining general traffic efficiency.

---

## Project Overview

- **Environment**: A realistic SUMO-based traffic simulation modeled on roads around Queen’s Medical Centre (QMC), Nottingham.
- **Goal**: Minimise emergency vehicle delays using intelligent, adaptive traffic signal control.
- **Frameworks**: Built using Python, SUMO, OpenAI Gym, and PyTorch.

---

## Reinforcement Learning Agents

| Agent     | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| **PPO**   | Proximal Policy Optimization (best-performing agent in this study).         |
| **A2C**   | Advantage Actor-Critic; struggled under sparse reward settings.             |
| **QMIX**  | Cooperative value-based MARL, coordinated across intersections.             |
| **Rule-Based** | Baseline controller with static pre-defined signal timings.           |

---

## Key Results

| Metric                       | PPO     | A2C     | QMIX    | Rule-Based |
|-----------------------------|---------|---------|---------|------------|
| **EV Waiting Time (s)**     | 0.40    | 607.01  | 20.27   | 5.70       |
| **EV Travel Time (s)**      | 10.37   | 60.35   | 23.53   | 473.90     |
| **Queue Length (vehicles)** | 0.00    | 0.04    | 0.01    | 0.02       |
| **Average Reward**          | 27.58   | -23.40  | 18.00   | 5.41       |

---

## Installation

```bash
git clone https://github.com/hamza09230923/OPTIMISING-URBAN-TRAFFIC-FOR-EMERGENCY-VEHICLE-PRIORITY-USING-MULTI-AGENT-REINFORCEMENT-LEARNING.git
cd traffic_marl_project
pip install -r requirements.txt
