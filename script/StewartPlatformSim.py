from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from StewartPlatform import *

# Python examples 
# https://github.com/CoppeliaRobotics/zmqRemoteApi/tree/master/clients/python 

client = RemoteAPIClient()

stewart_platform = StewartPlatform(client)
sim = client.require('sim')

# https://manual.coppeliarobotics.com/en/simulation.htm#stepped
sim.setStepping(True)

sim.startSimulation()

while (t := sim.getSimulationTime()) < 60:
    print(f'Simulation time: {t:.2f} [s]')

    stewart_platform.example_ik(t)

    sim.step()

sim.stopSimulation()