import math
import time
import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt

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


model_str='morelli'
integrator_str='rk45'
v2_integrators=False

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
    initial_state = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    
    # Define simulation parameters
    tmax = 100 # simulation time
    tmax_integrator = 400
    step = 1/30 # step time
    runtime = 0

    waypoints = []
    # Initial leader position
    leader_position = [[300, 300, alt]]
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

    while runtime < tmax:
        if integrator.status == 'running':
            while integrator.t < times[-1] + step:  # and "running" can be added
                integrator.step()
        else:
            print(f"Integrator status changed at: {runtime} seconds. Status: {integrator.status}")
            break

        dense_output = integrator.dense_output()

        t = times[-1] + step
            #print(f"{round(t, 2)} / {tmax}")

        leader_position[0][0] += 20
        leader_position[0][1] += 20
        ap.update_waypoints(leader_position)
        waypoints.append(copy.deepcopy(leader_position))

        times.append(t)
        states.append(dense_output(t))

        updated = ap.advance_discrete_mode(times[-1], states[-1])
        modes.append(ap.mode)

        print(f"Time: {t}, Mode: {ap.mode}, Status: {integrator.status}")

        if ap.is_finished(times[-1], states[-1]):
            # this both causes the outer loop to exit and sets res['status'] appropriately
            integrator.status = 'autopilot finished'
            print(f"Autopilot finished")
            break

        if updated:
            # re-initialize the integration class on discrete mode switches
            integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
            print(f"Simulation mode updated")
            break

        #if updated:
        #    # re-initialize the integration class on discrete mode switches
        #    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
        #    break

        # Accumulate runtime
        runtime += step

    res = {}
    res['status'] = integrator.status
    res['times'] = times
    res['states'] = np.array(states, dtype=float)
    res['modes'] = modes
    res['runtime'] = time.perf_counter() - start
    
    print(f"Simulation Completed in real-time mode for {runtime} seconds")
    
    plot_overhead(res, waypoints=waypoints)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    fig1 = plt.figure()
    ay = fig1.add_subplot(1,1,1)
    ay.plot(times, res['states'][:, StateIndex.VT])
    



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
            state = full_state[num_vars*i:num_vars*(i+1)]
            u_ref = u_refs[4*i:4*(i+1)]

            xd = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
            xds.append(xd)

        rv = np.hstack(xds)

        return rv
    
    return der_func

def plot_overhead(run_sim_result, waypoints=None, llc=None):
    '''altitude over time plot from run_f16_sum result object

    note: call plt.show() afterwards to have plot show up
    '''

    res = run_sim_result
    fig = plt.figure(figsize=(7, 5))

    ax = fig.add_subplot(1, 1, 1)

    full_states = res['states']


    ys = full_states[:, StateIndex.POSN] # 9: n/s position (ft)
    xs = full_states[:, StateIndex.POSE] # 10: e/w position (ft)

    ax.plot(xs, ys, '-', color='blue', linewidth=10, label='Path')

    ax.plot([xs[0]], [ys[1]], 'k*', ms=8, label='Start')

    if waypoints is not None:
        xs = [wp[0][0] for wp in waypoints]
        ys = [wp[0][1] for wp in waypoints]

        ax.plot(xs, ys, 'ro', linewidth=0.1, ms=4, label='Waypoints')

    ax.set_ylabel('North / South Position (ft)')
    ax.set_xlabel('East / West Position (ft)')
    
    ax.set_title('Overhead Plot')
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()


if __name__ == '__main__':
    main()
