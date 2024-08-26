#!/usr/bin/env python3
import rospy
from ekfspecific import EKFSpecific

def main():
    rospy.init_node("stewart_platform_ekf")
    print('start')

    stewart_platform_ekf = EKFSpecific()

    while (t := stewart_platform_ekf.sim.getSimulationTime()) < 10:
        stewart_platform_ekf.update()

    stewart_platform_ekf.stop_simulation()
    stewart_platform_ekf.plot_data()

if __name__ == '__main__':
    main()
