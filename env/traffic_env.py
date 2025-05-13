import gym
import numpy as np
import traci
from gym import spaces

class TrafficEnv(gym.Env):
    def __init__(self, sumo_cfg="osm.sumocfg", use_gui=False, max_steps=1800):
        super(TrafficEnv, self).__init__()
        self.sumo_cfg = sumo_cfg
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.step_count = 0

        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.sumo_cmd = [self.sumo_binary, "-c", self.sumo_cfg, "--start"]

        traci.start(self.sumo_cmd)

        self.traffic_lights = traci.trafficlight.getIDList()
        self.lane_ids = traci.lane.getIDList()

        self.num_lanes = len(self.lane_ids)
        self.num_tls = len(self.traffic_lights)

        self.observation_space = spaces.Box(
            low=0, high=100, shape=(self.num_lanes,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([4] * self.num_tls)

        self.prev_phases = {tl: -1 for tl in self.traffic_lights}
        self.same_phase_count = {tl: 0 for tl in self.traffic_lights}

    def reset(self, seed=None, options=None):
        self.step_count = 0
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)
        self.same_phase_count = {tl: 0 for tl in self.traffic_lights}
        for _ in range(10):
            traci.simulationStep()
        return self._get_state(), {}

    def step(self, actions):
        self.step_count += 1
        for tl_id, action in zip(self.traffic_lights, actions):
            traci.trafficlight.setPhase(tl_id, action)

        traci.simulationStep()
        self._control_emergency_vehicles()
        self._track_phase_changes()

        obs = self._get_state()
        reward = self._compute_reward()
        done = self.step_count >= self.max_steps or traci.simulation.getMinExpectedNumber() == 0
        if done and traci.simulation.getMinExpectedNumber() == 0:
            reward += 10
        info = {}

        return obs, reward, done, False, info

    def _get_state(self):
        return np.array([
            traci.lane.getLastStepVehicleNumber(lane_id)
            for lane_id in self.lane_ids
        ], dtype=np.float32)

    def _compute_reward(self):
        ev_wait = self.get_total_ev_waiting_time()
        ev_wait_norm = ev_wait / (len(traci.vehicle.getIDList()) + 1)

        ev_stop_penalty = self._get_ev_stop_penalty()
        ev_speed_bonus = self._get_average_ev_speed_bonus()
        ev_move_bonus = self._get_ev_movement_bonus()
        green_bonus = self._get_green_for_ev_bonus()
        anticipation_bonus = self._get_ev_anticipation_bonus()
        red_hold_penalty = self._get_red_hold_penalty()
        phase_penalty = sum(1 for count in self.same_phase_count.values() if count > 5)

        reward = (
            - 5.0 * ev_wait_norm
            - 7.0 * ev_stop_penalty
            - 5.0 * red_hold_penalty
            + 3.0 * ev_speed_bonus
            + 2.0 * ev_move_bonus
            + 4.0 * green_bonus
            + 3.0 * anticipation_bonus
            - 4.0 * phase_penalty
        )

        reward = np.clip(reward, -30, 30)

        print(f"[Step {self.step_count}] R: {reward:.2f} | EVWait: {ev_wait:.1f}, StopPen: {ev_stop_penalty}, "
              f"HoldPen: {red_hold_penalty}, Antic: {anticipation_bonus}, Speed+: {ev_speed_bonus:.2f}, "
              f"Move+: {ev_move_bonus}, Green+: {green_bonus}")

        return reward

    def _control_emergency_vehicles(self):
        for veh_id in traci.vehicle.getIDList():
            if self._is_ev(veh_id):
                try:
                    dist_to_intersection = traci.vehicle.getLanePosition(veh_id)
                    lane_id = traci.vehicle.getLaneID(veh_id)
                    for tl in self.traffic_lights:
                        controlled_lanes = traci.trafficlight.getControlledLanes(tl)
                        if lane_id in controlled_lanes and dist_to_intersection < 100:
                            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0]
                            green_phases = [i for i, p in enumerate(logic.phases) if 'G' in p.state]
                            if green_phases:
                                traci.trafficlight.setPhase(tl, green_phases[0])

                    if self.step_count % 5 == 0:
                        traci.vehicle.rerouteTraveltime(veh_id)
                    traci.vehicle.setSpeed(veh_id, traci.vehicle.getAllowedSpeed(veh_id))
                    traci.vehicle.setLaneChangeMode(veh_id, 0b011001011011)
                except traci.TraCIException:
                    pass

    def _track_phase_changes(self):
        for tl in self.traffic_lights:
            current = traci.trafficlight.getPhase(tl)
            if current == self.prev_phases[tl]:
                self.same_phase_count[tl] += 1
            else:
                self.same_phase_count[tl] = 0
            self.prev_phases[tl] = current

    def _get_ev_stop_penalty(self):
        penalty = 0
        for veh_id in traci.vehicle.getIDList():
            if self._is_ev(veh_id):
                try:
                    if traci.vehicle.getSpeed(veh_id) < 0.1:
                        penalty += 1
                except traci.TraCIException:
                    continue
        return penalty

    def _get_ev_movement_bonus(self):
        bonus = 0
        for veh_id in traci.vehicle.getIDList():
            if self._is_ev(veh_id):
                try:
                    if traci.vehicle.getSpeed(veh_id) > 0.1:
                        bonus += 1
                except traci.TraCIException:
                    continue
        return bonus

    def get_average_ev_travel_time(self):
        ev_ids = [veh_id for veh_id in traci.vehicle.getIDList() if self._is_ev(veh_id)]
        travel_times = []
        for veh_id in ev_ids:
            try:
                travel_times.append(traci.vehicle.getAccumulatedWaitingTime(veh_id) + traci.vehicle.getSpeed(veh_id))
            except traci.TraCIException:
                continue
        return np.mean(travel_times) if travel_times else 0.0

    def _get_green_for_ev_bonus(self):
        bonus = 0
        for veh_id in traci.vehicle.getIDList():
            if self._is_ev(veh_id):
                try:
                    lane_id = traci.vehicle.getLaneID(veh_id)
                    for tl in self.traffic_lights:
                        controlled_lanes = traci.trafficlight.getControlledLanes(tl)
                        if lane_id in controlled_lanes:
                            phase = traci.trafficlight.getPhase(tl)
                            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0]
                            if 'G' in logic.phases[phase].state:
                                bonus += 1
                except traci.TraCIException:
                    continue
        return bonus

    def _get_ev_anticipation_bonus(self):
        bonus = 0
        for veh_id in traci.vehicle.getIDList():
            if self._is_ev(veh_id):
                try:
                    if traci.vehicle.getLanePosition(veh_id) > 50:  # moved anticipation earlier
                        bonus += 1
                except traci.TraCIException:
                    continue
        return bonus

    def _get_red_hold_penalty(self):
        penalty = 0
        for veh_id in traci.vehicle.getIDList():
            if self._is_ev(veh_id):
                try:
                    if traci.vehicle.getSpeed(veh_id) < 0.1 and traci.vehicle.getLanePosition(veh_id) > 80:
                        penalty += 1
                except traci.TraCIException:
                    continue
        return penalty

    def _get_average_ev_speed_bonus(self):
        speeds = []
        for veh_id in traci.vehicle.getIDList():
            if self._is_ev(veh_id):
                try:
                    speeds.append(traci.vehicle.getSpeed(veh_id))
                except traci.TraCIException:
                    continue
        return sum(speeds) / len(speeds) if speeds else 0.0

    def _is_ev(self, veh_id):
        try:
            return traci.vehicle.getTypeID(veh_id) == "bus_bus"
        except traci.TraCIException:
            return False

    def get_total_ev_waiting_time(self):
        ev_ids = [veh_id for veh_id in traci.vehicle.getIDList() if self._is_ev(veh_id)]
        return sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in ev_ids)

    def get_average_queue_length(self):
        total_halted = sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in self.lane_ids)
        return total_halted / len(self.lane_ids) if self.lane_ids else 0.0

    def close(self):
        traci.close()


# TALK ABOUT GYMNASIUEM ISSUE