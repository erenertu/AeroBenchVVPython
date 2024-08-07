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
integrator_str = 'euler'
v2_integrators = True


def make_der_func(model_str, v2_integrators, u_deg):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'

        xd = controlled_f16(full_state, u_deg, model_str, v2_integrators)[0]
        rv = xd

        return rv

    return der_func


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


# Define simulation parameters
step = 1 / 30  # step time
tmax = (10) * (1/step)  # simulation time
tmax_integrator = 0
runtime = 0


# Build Initial Condition Vectors
# state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
initial_state_f = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

start = time.perf_counter()

initial_state_f = np.array(initial_state_f, dtype=float)
x0 = initial_state_f

# run the numerical simulation
times = [0]
states = [x0]
controls = [[1, 0, 0, 0]]

der_func = make_der_func(model_str, v2_integrators, controls[-1])

if integrator_str == 'rk45':
    integrator_class = RK45
    kwargs = {}
else:
    assert integrator_str == 'euler'
    integrator_class = Euler
    kwargs = {'step': step}

# note: fixed_step argument is unused by rk45, used with euler
integrator = integrator_class(der_func, times[-1], states[-1], **kwargs)

# Simulation part
while runtime < tmax:
    loop_start_time = time.perf_counter()
    # FDM update and states of the follower
    integrator.step()
    dense_output = integrator.dense_output()
    t = times[-1] + step
    times.append(t)
    states.append(dense_output(t))
    joystick_inputs = [0.8, 0, 0, 0]
    controls.append(joystick_inputs)
    #print(f"Time: {t * step}, Mode: {ap.mode}, Status: {integrator.status}")
    print('alt: ', states[-1][11])

    runtime += step
    fps_recorder = time.perf_counter() - loop_start_time
    #print('FPS:', fps_recorder * 0.001)
    # Synchronize with real time

north = [state[9] for state in states]
east = [state[10] for state in states]
altitude = [state[11] for state in states]
vt = [state[0] for state in states]
heading = [state[5] for state in states]

# Plot PID data
fig = plt.figure()
ax1 = fig.add_subplot(311)
##### Part for heading
ax1.plot(times, heading, label='Heading')
##### Part for velo
ax1.plot(times, vt, label='Velocity')
ax1.set_xlabel('Time (s)')
ax1.legend()
ax2 = fig.add_subplot(312)
ax2.plot(times, altitude, label='Altitude', color='green')
##### Part for velo PID
ax2.set_title('Altitude - Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Values')
ax2.legend()
ax3 = fig.add_subplot(313)
ax3.plot(times, east, marker='x', color='red', label='East', alpha=0.1)
ax3.plot(times, north, label='North')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('East / North')
ax3.legend()
plt.tight_layout()
plt.show()