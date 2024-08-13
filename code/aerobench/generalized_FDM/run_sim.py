import math
import time
import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from mpl_toolkits.mplot3d import Axes3D, art3d
import pygame
import asyncio

import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(path)
os.chdir(path)
import copy
from simple_pid import PID
from scipy.integrate import RK45

from fdm_config import fdm_config
from util import get_state_names, Euler, StateIndex

model_str = 'morelli_f16'
integrator_str = 'euler'
v2_integrators = True


def make_der_func(model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'

        # Get joystick inputs
        u_deg = get_joystick_inputs() # TODO: Joystick inputs should be generalised, not directly deflection values

        # Call flight dynamics model config of the desired aircraft
        xd = fdm_config(full_state, u_deg, model_str, v2_integrators)
        rv = xd

        return rv

    return der_func

def get_joystick_inputs():
    pygame.event.pump()  # Process event queue

    # Get joystick axes (example for 4 axes)
    axis_0 = joystick.get_axis(0) * 21.5  # Aileron (roll)
    axis_1 = joystick.get_axis(1) * 25  # Elevator (pitch)
    axis_2 = joystick.get_axis(3)  # Throttle
    axis_3 = joystick.get_axis(2) * 30  # Rudder (yaw)

    # Map joystick axes to control inputs
    aileron = -axis_0
    elevator = -axis_1
    throttle = (1 - axis_2) / 2  # Map from [-1, 1] to [0, 1]
    rudder = -axis_3

    return [throttle, elevator, aileron, rudder]


### Initial Conditions ###
# TODO: Check if we should give NED inital values or not

# Parse the XML file
tree = ET.parse('init_cond.xml')
root = tree.getroot()

power = float(root.find('power').text)  # engine power level (0-10)
# Default alpha & beta
alpha = float(root.find('alpha').text)  # Trim Angle of Attack (rad)
beta = float(root.find('beta').text)  # Side slip angle (rad)

# Initial Attitude
alt = float(root.find('altitude').text)  # altitude (ft)
vt = float(root.find('velocity').text)  # initial velocity (ft/sec)

phi = float(root.find('phi').text)  # Roll angle from wings level (rad)
theta = float(root.find('theta').text)  # Pitch angle from nose level (rad)
psi = float(root.find('psi').text)  # Yaw angle from North (rad)


# Define simulation parameters
step = 1 / 30  # step time

# Build Initial Condition Vectors
# state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
initial_state_f = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

start = time.perf_counter()
x0 = np.array(initial_state_f, dtype=float)

# run the numerical simulation
times = [0]
states = [x0]

der_func = make_der_func(model_str, v2_integrators)

if integrator_str == 'rk45':
    integrator_class = RK45
    kwargs = {}
else:
    assert integrator_str == 'euler'
    integrator_class = Euler
    kwargs = {'step': step}

# note: fixed_step argument is unused by rk45, used with euler
integrator = integrator_class(der_func, times[-1], states[-1], **kwargs)

# Initialize pygame and the joystick
pygame.init()
pygame.joystick.init()

# Assuming there is at least one joystick connected
try:
    # To check if joystick connected
    joystick = pygame.joystick.Joystick(0)
    print(f"Initialized Joystick: {joystick.get_name()}")
except:
    # Code to handle any exception
    print("No joystick connected.")
    sys.exit()

joystick.init()

# Plotting Part
# Initialize matplotlib for real-time 3D plotting
x, y, z = 0, 0, alt  # initial positions
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
point, = ax.plot([x], [y], [z], 'bo')  # 'bo' stands for blue circle
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(0, 2000)
ax.invert_yaxis()
ax.set_xlabel('North Position (ft)')
ax.set_ylabel('East Position (ft)')
ax.set_zlabel('Altitude (ft)')
ax.set_title('Real-Time 3D Flight Dynamics Simulation')

# Create lines for the x, y, and z axes
line_x, = ax.plot([], [], [], 'r--', label='x-axis')
line_y, = ax.plot([], [], [], 'g--', label='y-axis')
line_z, = ax.plot([], [], [], 'b--', label='z-axis')


# Annotations for forces and velocity
text_phi = ax.text2D(0.05, 0.90, '', transform=ax.transAxes)
text_theta = ax.text2D(0.05, 0.85, '', transform=ax.transAxes)
text_psi = ax.text2D(0.05, 0.80, '', transform=ax.transAxes)
text_velocity = ax.text2D(0.05, 0.75, '', transform=ax.transAxes)
def update_orientation_lines(ax, state, scale=200):
    phi = state[StateIndex.PHI]
    theta = state[StateIndex.THETA]
    psi = state[StateIndex.PSI]

    # Rotation matrix
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    cos_psi, sin_psi = math.cos(psi), math.sin(psi)

    R = np.array([
        [cos_theta * cos_psi, cos_theta * sin_psi, -sin_theta],
        [sin_phi * sin_theta * cos_psi - cos_phi * sin_psi, sin_phi * sin_theta * sin_psi + cos_phi * cos_psi, sin_phi * cos_theta],
        [cos_phi * sin_theta * cos_psi + sin_phi * sin_psi, cos_phi * sin_theta * sin_psi - sin_phi * cos_psi, cos_phi * cos_theta]
    ])

    origin = np.array([state[9], state[10], state[11]])

    # Define the unit vectors for the aircraft's local axes
    x_axis = np.array([scale, 0, 0])
    y_axis = np.array([0, scale, 0])
    z_axis = np.array([0, 0, scale])

    # Rotate the unit vectors
    x_axis_rot = R @ x_axis
    y_axis_rot = R @ y_axis
    z_axis_rot = R @ z_axis

    # Update the lines
    line_x.set_data([origin[0], origin[0] + x_axis_rot[0]], [origin[1], origin[1] + x_axis_rot[1]])
    line_x.set_3d_properties([origin[2], origin[2] + x_axis_rot[2]])

    line_y.set_data([origin[0], origin[0] + y_axis_rot[0]], [origin[1], origin[1] + y_axis_rot[1]])
    line_y.set_3d_properties([origin[2], origin[2] + y_axis_rot[2]])

    line_z.set_data([origin[0], origin[0] + z_axis_rot[0]], [origin[1], origin[1] + z_axis_rot[1]])
    line_z.set_3d_properties([origin[2], origin[2] + z_axis_rot[2]])

# Simulation part
while True:
    loop_start_time = time.time()

    # FDM update and states of the follower
    integrator.step()
    dense_output = integrator.dense_output()
    t = times[-1] + step
    times.append(t)
    states.append(dense_output(t))

    # Update plot
    point.set_data([states[-1][9]], [states[-1][10]])
    point.set_3d_properties([states[-1][11]])
    ax.set_xlim(states[-1][9] - 1000, states[-1][9] + 1000)  # Center the plot around the current x position
    ax.set_ylim(states[-1][10] - 1000, states[-1][10] + 1000)  # Center the plot around the current y position
    ax.set_zlim(states[-1][11] - 1000, states[-1][11] + 1000)
    ax.invert_yaxis()

    # Update orientation lines
    update_orientation_lines(ax, states[-1])

    # Update annotations
    text_phi.set_text(f'Phi: {math.degrees(states[-1][3]):.2f} deg')
    text_theta.set_text(f'Theta: {math.degrees(states[-1][4]):.2f} deg')
    text_psi.set_text(f'Psi: {math.degrees(states[-1][5]):.2f} deg')
    text_velocity.set_text(f'Velocity: {states[-1][0]:.2f} ft/s')

    fig.canvas.draw()
    fig.canvas.flush_events()

    # Synchronize with real time
    elapsed_time = time.time() - loop_start_time
    #print('Compute Time: ', elapsed_time * 1000)
    sleep_time = max(0, step - elapsed_time)
    time.sleep(sleep_time)

# Quit pygame
pygame.quit()

# Keep the plot open after the simulation ends
plt.ioff()
plt.show()