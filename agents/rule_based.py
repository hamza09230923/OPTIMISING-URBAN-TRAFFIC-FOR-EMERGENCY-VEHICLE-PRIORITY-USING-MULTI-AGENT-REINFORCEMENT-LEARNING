# agents/rule_based.py
import traci


def run_rule_based(sumocfg_path):
    if traci.isLoaded():
        traci.close()
    traci.start(["sumo", "-c", sumocfg_path])

    junction_id = traci.trafficlight.getIDList()[0]
    step = 0

    ev_ids = set()
    ev_wait_times = {}
    ev_start_times = {}

    while traci.simulation.getMinExpectedNumber() > 0 and step < 1000:
        if step % 30 == 0:
            current_phase = traci.trafficlight.getPhase(junction_id)
            phases = traci.trafficlight.getAllProgramLogics(junction_id)[0].phases
            num_phases = len(phases)
            traci.trafficlight.setPhase(junction_id, (current_phase + 1) % num_phases)

        for vid in traci.vehicle.getIDList():
            vtype = traci.vehicle.getTypeID(vid)
            if "bus" in vtype:  # matches "bus_bus", "bus1", etc.
                ev_ids.add(vid)
                if vid not in ev_start_times:
                    ev_start_times[vid] = traci.simulation.getTime()
                ev_wait_times[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)

        traci.simulationStep()
        step += 1

    traci.close()

    ev_metrics = {
        ev: {
            "travel_time": step - ev_start_times.get(ev, 0),
            "wait_time": ev_wait_times.get(ev, 0)
        } for ev in ev_ids
    }

    return ev_metrics

