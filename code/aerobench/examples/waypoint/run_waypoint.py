import math
import time
import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import asyncio

import sys
import os
import copy

# Add the aerobench directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from aerobench.visualize import plot

from waypoint_autopilot import WaypointAutopilot
from scipy.integrate import RK45

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.util import get_state_names, Euler, StateIndex

model_str = 'morelli'
integrator_str = 'rk45'
v2_integrators = False

def make_der_func(ap, model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'

        u_refs = ap.get_checked_u_ref(t, full_state)

        num_aircraft = u_refs.size // 4
        num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
        assert full_state.size // num_vars == num_aircraft

        xds = []

        for i in range(num_aircraft):
            state = full_state[num_vars * i:num_vars * (i + 1)]
            u_ref = u_refs[4 * i:4 * (i + 1)]

            xd = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
            xds.append(xd)

        rv = np.hstack(xds)

        return rv

    return der_func

async def main():
    'main function'

    ### Initial Conditions ###
    power = 9  # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215)  # Trim Angle of Attack (rad)
    beta = 0  # Side slip angle (rad)

    # Initial Attitude
    alt = 3800  # altitude (ft)
    vt = 540  # initial velocity (ft/sec)
    phi = 0  # Roll angle from wings level (rad)
    theta = 0  # Pitch angle from nose level (rad)
    psi = math.pi / 8  # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initial_state = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    # Define simulation parameters
    tmax = 100  # simulation time
    tmax_integrator = 400
    step = 1 / 30  # step time
    runtime = 0

    waypoints = []
    # Initial leader position
    leader_position = [[3000, 3000, alt]]
    waypoints.append(copy.deepcopy(leader_position))

    ap = WaypointAutopilot(leader_position, stdout=True)

    start = time.perf_counter()

    initial_state = np.array(initial_state, dtype=float)
    llc = ap.llc

    num_vars = len(get_state_names()) + llc.get_num_integrators()

    if initial_state.size < num_vars:
        # append integral error states to state vector
        x0 = np.zeros(num_vars)
        x0[:initial_state.shape[0]] = initial_state
    else:
        x0 = initial_state

    assert x0.size % num_vars == 0, f"expected initial state ({x0.size} vars) to be multiple of {num_vars} vars"

    # run the numerical simulation
    times = [0]
    states = [x0]

    # mode can change at time 0
    ap.advance_discrete_mode(times[-1], states[-1])

    modes = [ap.mode]

    der_func = make_der_func(ap, model_str, v2_integrators)

    if integrator_str == 'rk45':
        integrator_class = RK45
        kwargs = {}
    else:
        assert integrator_str == 'euler'
        integrator_class = Euler
        kwargs = {'step': step}

    # note: fixed_step argument is unused by rk45, used with euler
    integrator = integrator_class(der_func, times[-1], states[-1], tmax_integrator, **kwargs)

    fig, ax = plt.subplots(figsize=(7, 5))
    xs, ys = [], []
    waypoint_xs, waypoint_ys = [], []
    line, = ax.plot([], [], '-', color='blue', linewidth=2, label='Path')
    waypoint_line, = ax.plot([], [], 'ro', linewidth=0.1, ms=4, label='Waypoints')

    velocity_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    psi_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    
    def init():
        ax.set_xlim(-10000, 10000)
        ax.set_ylim(-10000, 10000)
        ax.set_ylabel('North / South Position (ft)')
        ax.set_xlabel('East / West Position (ft)')
        ax.set_title('Overhead Plot')
        ax.legend()
        return line, waypoint_line, velocity_text, psi_text

    def update_plot(frame):
        
        nonlocal runtime, integrator, leader_position

        if integrator.status == 'running':
            while integrator.t < times[-1] + step:
                integrator.step()
        else:
            print(f"Integrator status changed at: {runtime} seconds. Status: {integrator.status}")
            return line, waypoint_line, velocity_text, psi_text

        dense_output = integrator.dense_output()

        t = times[-1] + step

        leader_position[0][0] += 250 * step
        leader_position[0][1] += 250 * step
        ap.update_waypoints(leader_position)
        waypoints.append(copy.deepcopy(leader_position))

        times.append(t)
        states.append(dense_output(t))

        updated = ap.advance_discrete_mode(times[-1], states[-1])
        modes.append(ap.mode)

        print(f"Time: {t}, Mode: {ap.mode}, Status: {integrator.status}")

        if ap.is_finished(times[-1], states[-1]):
            integrator.status = 'autopilot finished'
            print(f"Autopilot finished")
            return line, waypoint_line, velocity_text, psi_text

        if updated:
            integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
            print(f"Simulation mode updated")
            return line, waypoint_line, velocity_text, psi_text

        xs.append(states[-1][StateIndex.POSE])
        ys.append(states[-1][StateIndex.POSN])

        line.set_data(xs, ys)

        waypoint_xs.append(leader_position[0][0])
        waypoint_ys.append(leader_position[0][1])
        waypoint_line.set_data(waypoint_xs, waypoint_ys)

        # Center the plot around the current position
        ax.set_xlim(states[-1][StateIndex.POSE] - 1000, states[-1][StateIndex.POSE] + 1000)
        ax.set_ylim(states[-1][StateIndex.POSN] - 1000, states[-1][StateIndex.POSN] + 1000)

        velocity_text.set_text(f'Velocity: {states[-1][StateIndex.VT]:.2f} ft/s')
        psi_text.set_text(f'Psi: {states[-1][StateIndex.PSI]:.2f} rad')

        runtime += step
        return line, waypoint_line, velocity_text, psi_text

    ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0, tmax, step), init_func=init, blit=True, interval=1000 * step)
    plt.show()

    while runtime < tmax:
        loop_start_time = time.time()
        update_plot(None)
        time.sleep()
        await asyncio.sleep(max(0, step - (time.time() - loop_start_time)))

    print(f"Simulation Completed in real-time mode for {runtime} seconds")


if __name__ == '__main__':
    asyncio.run(main())
