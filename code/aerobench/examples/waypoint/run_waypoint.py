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
from simple_pid import PID

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
    
def formation_navigator(states_F, states_L, initial_state_F, initial_state_L, clearances):
    'states = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]'
    heading_diffs = states_F[5] - states_L[5]  # psi values in radian for both leader and follower
    #initial_NE_diffs = [initial_state_F[9] - initial_state_L[9], initial_state_F[10] - initial_state_L[10]]
    initial_NE_diffs = [0, 0]
    states_NE_diffs = [states_L[9] - states_F[9], states_L[10] - states_F[10]]

    clearence_transform = np.dot(
                                np.array([[math.cos(heading_diffs), math.sin(heading_diffs)],
                                    [-math.sin(heading_diffs), math.cos(heading_diffs)]]),
                                np.array([[clearances[1]], [clearances[0]]])
                                ) 
    clearences_f = clearence_transform - np.array([[initial_state_F[9]], [initial_state_F[10]]])

    clearances_err = np.dot(
                            np.array([[math.cos(states_F[5]), math.sin(states_F[5])], [-math.sin(states_F[5]), math.cos(states_F[5])]]),
                            np.array([[states_NE_diffs[0] + initial_NE_diffs[0] + clearences_f[0][0]], [states_NE_diffs[1] + initial_NE_diffs[1] + clearences_f[1][0]]])
                            )

    # PID algorithm there
    pid_heading = PID(0.00012, 0, 0, setpoint=0)
    pid_heading.output_limits = (-math.radians(45), math.radians(45))
    delta_heading = pid_heading(-clearances_err[1])
    pid_velo = PID(0.0214, 0.05, 0, setpoint=0)
    delta_velo = pid_velo(-clearances_err[0])
    #delta_velo = float(-100)

    return float(delta_heading), delta_velo[0], clearances_err[1][0], pid_heading, clearances_err[0][0], pid_velo


### Initial Conditions ###
power = 5  # engine power level (0-10)
# Default alpha & beta
alpha = deg2rad(0)  # Trim Angle of Attack (rad)
beta = 0  # Side slip angle (rad)

# Initial Attitude
alt = 1500  # altitude (ft)
vt = 400  # initial velocity (ft/sec)

phi = 0  # Roll angle from wings level (rad)
theta = 0  # Pitch angle from nose level (rad)
psi = 0  # Yaw angle from North (rad)


# Formation clearances according to the formation type
clearances = [0, 150]  # east and north clearances respectively

# Define simulation parameters
step = 1 / 30  # step time
tmax = (10) * (1/step)  # simulation time
tmax_integrator = 4000
runtime = 0

waypoints = []
# Initial leader position
leader_position = [[0, 5000, alt]]  # east, north and altitude
target_position = [[leader_position[0][0] + clearances[0], leader_position[0][1] + clearances[1], alt]]
waypoints.append(copy.deepcopy(target_position))

ap = WaypointAutopilot(target_position, stdout=True)

# Build Initial Condition Vectors
# state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
initial_state_f = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
# Leader
v_north_init_leader = 400
v_east_init_leader = 400
vt_leader = math.sqrt(v_east_init_leader ** 2 + v_north_init_leader ** 2)
leader_psi = math.atan2(v_north_init_leader, v_east_init_leader)
if leader_psi < 0:
    leader_psi += 2 * math.pi
initial_state_l = [vt_leader, 0, 0, 0, 0, leader_psi, 0, 0, 0, leader_position[0][1], leader_position[0][0], alt, 0]

start = time.perf_counter()

initial_state_f = np.array(initial_state_f, dtype=float)
llc = ap.llc

num_vars = len(get_state_names()) + llc.get_num_integrators()

if initial_state_f.size < num_vars:
    # append integral error states to state vector
    x0 = np.zeros(num_vars)
    x0[:initial_state_f.shape[0]] = initial_state_f
else:
    x0 = initial_state_f

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

# Initialize lists for PID data
pid_times = []
heading_setpoints = []
velo_setpoints = []
lateral_errors = []
forward_errors = []
des_head_array = []
des_velo_array = []
lead_heading = []

v_east_leader = v_east_init_leader
v_north_leader = v_north_init_leader

# Simulation part
while runtime < tmax:
    loop_start_time = time.perf_counter()
    # FDM update and states of the follower
    if integrator.status == 'running':
        while integrator.t < times[-1] + step:
            integrator.step()
    else:
        print(f"Integrator status changed at: {runtime} seconds. Status: {integrator.status}")
    dense_output = integrator.dense_output()
    t = times[-1] + step
    times.append(t)
    states.append(dense_output(t))
    
   
    if t * step > 3 and t * step < 5:
        v_east_leader = v_east_leader + 5 * step
        v_north_leader = v_north_leader 
    elif t * step < 3:
        v_east_leader = v_east_init_leader
        v_north_leader = v_north_init_leader
    elif t * step > 8 and t * step < 12:
        v_east_leader = v_east_leader
        v_north_leader = v_north_leader - 1 * step

    vt_leader = math.sqrt(v_east_leader ** 2 + v_north_leader ** 2)
    
     # Update of leader aircraft and target positin
    leader_position[0][0] += v_east_leader * step  # east   
    leader_position[0][1] += v_north_leader * step  # north
    leader_psi = math.atan2( v_east_leader, v_north_leader)
    if leader_psi < 0:
        leader_psi += 2 * math.pi
    states_leader = [0, 0, 0, 0, 0, leader_psi, 0, 0, 0, leader_position[0][1], leader_position[0][0], alt, 0]

    target_position = [[leader_position[0][0] + clearances[0], leader_position[0][1] + clearances[1], alt]]  # east, north and altitude
    ap.update_waypoints(target_position)
    waypoints.append(copy.deepcopy(target_position))

    # Formation navigator and setting to the autopilot
    delta_heading, delta_velo, lateral_error, pid_heading, forward_error, pid_velo = formation_navigator(states[-1], states_leader, initial_state_f, initial_state_l, clearances)
    des_head = states_leader[5] + delta_heading
    des_velo = vt_leader + delta_velo
    if des_velo > 1000:
        des_velo = 1000
    elif des_velo < 250:
        des_velo = 250
    ap.get_waypoint_data_pid(des_head, des_velo)
    
    updated = ap.advance_discrete_mode(times[-1], states[-1])
    modes.append(ap.mode)
    print(f"Time: {t * step}, Mode: {ap.mode}, Status: {integrator.status}")

    if ap.is_finished(times[-1], states[-1]):
        integrator.status = 'autopilot finished'
        print(f"Autopilot finished")

    if updated:
        integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
        print(f"Simulation mode updated")

    #############################################
    # Store PID data
    pid_times.append(t * step)
    heading_setpoints.append(pid_heading.setpoint)
    velo_setpoints.append(pid_velo.setpoint)
    lateral_errors.append(lateral_error)
    forward_errors.append(forward_error)
    #
    des_head_array.append(delta_heading)
    lead_heading.append(states_leader[5])
    des_velo_array.append(vt_leader)
    ############################################

    runtime += step
    fps_recorder = time.perf_counter() - loop_start_time
    #print('FPS:', fps_recorder * 0.001)
    # Synchronize with real time

north = [state[9] for state in states]
east = [state[10] for state in states]
vt = [state[0] for state in states]
heading = [state[5] for state in states]
waypoint_north = [wp[0][1] for wp in waypoints]
waypoint_east = [wp[0][0] for wp in waypoints]

# Plot PID data
fig = plt.figure()
ax1 = fig.add_subplot(311)
##### Part for heading PID
ax1.plot(pid_times, des_head_array, label='Delta Heading')
ax1.plot(pid_times, heading[:-1], label='Heading')
ax1.plot(pid_times, lead_heading, label='Leader Heading')
##### Part for velo PID
#ax1.plot(pid_times, des_velo_array, label='Leader Velocity')
#ax1.plot(pid_times, vt[:-1], label='Follower Velocity')
ax1.set_xlabel('Time (s)')
ax1.legend()
ax2 = fig.add_subplot(312)
ax2.plot(pid_times, heading_setpoints, label='Setpoint', color='green')
##### Part for velo PID
#ax2.plot(pid_times, forward_errors, label='Forward Error', color='red')
##### Part for heading PID
ax2.plot(pid_times, lateral_errors, label='Lateral Error', color='red')
ax2.set_title('PID Controller')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Values')
ax2.legend()
ax3 = fig.add_subplot(313)
ax3.plot(waypoint_north, waypoint_east, marker='x', color='red', label='Target Point', alpha=0.1)
ax3.plot(north, east, label='Follower Aircraft')
ax3.set_xlabel('North (ft)')
ax3.set_ylabel('East (east)')
ax3.legend()
plt.tight_layout()
plt.show()