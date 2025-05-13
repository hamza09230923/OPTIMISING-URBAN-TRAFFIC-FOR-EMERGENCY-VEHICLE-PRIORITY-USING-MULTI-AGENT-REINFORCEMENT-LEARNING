import traci
import sumolib

# Path to your .sumocfg file sumo_cfg = "C:/Users/habdi/traffic_marl_project/sumo/osm.sumocfg"
sumo_binary = "sumo"  # or "sumo-gui" if you want GUI

traci.start([sumo_binary, "-c", sumo_cfg, "--start"])
traffic_lights = traci.trafficlight.getIDList()
print(f"🚦 Total traffic lights: {len(traffic_lights)}")
print(f"IDs: {traffic_lights}")
traci.close()
