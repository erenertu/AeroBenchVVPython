import math
import time
import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt

import sys
import os

# Add the aerobench directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from aerobench.run_f16_sim import run_f16_sim
from aerobench.visualize import plot

from waypoint_autopilot import WaypointAutopilot

def main():
    'main function'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 3800        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = math.pi/8   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    
    # Define simulation parameters
    tmax = 700 # simulation time
    step = 1/30
    extended_states = True

    # Initial target position offset
    offset_x = 100 / 0.3048  # Convert meters to feet
    offset_y = 100 / 0.3048  # Convert meters to feet

    # Initial leader position
    leader_position = np.array([0, 0, alt])

    ap = WaypointAutopilot([], stdout=True)

    # Real-time simulation loop
    runtime = 0
    while runtime < tmax:
        # Update leader position (for demonstration, we'll just move it linearly)
        leader_position += np.array([1, 1, 0])  # Simple linear movement

        # Compute new waypoint based on leader position
        e_pt = leader_position[0] + offset_x
        n_pt = leader_position[1] + offset_y
        h_pt = leader_position[2]

        waypoints = [[e_pt, n_pt, h_pt]]
        ap.update_waypoints(waypoints)

        # Run the simulation for one step
        res = run_f16_sim(init, step, ap, step=step, extended_states=extended_states, integrator_str='rk45')

        # Update initial conditions for the next iteration
        init = res['states'][-1]

        # Sleep to simulate real-time (adjust sleep duration to match real-time step)
        time.sleep(step)

        # Accumulate runtime
        runtime += step

    print(f"Simulation Completed in real-time mode for {tmax} seconds")

    # Plot results
    plot.plot_single(res, 'alt', title='Altitude (ft)')
    filename = 'alt.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_overhead(res, waypoints=waypoints)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_attitude(res)
    filename = 'attitude.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot inner loop controls + references
    plot.plot_inner_loop(res)
    filename = 'inner_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot outer loop controls + references
    plot.plot_outer_loop(res)
    filename = 'outer_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

if __name__ == '__main__':
    main()
