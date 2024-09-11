import os  # noqa: I001
import time
import math
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jsbsim
from jsbsim import FGFDMExec


from numpy import deg2rad

# Add the aerobench directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from aerobench.visualize import plot

from waypoint_autopilot import WaypointAutopilot

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.util import get_state_names, Euler, StateIndex

model_str = 'morelli'
integrator_str = 'euler'
v2_integrators = False

FT_TO_M = 0.3048

# Add the aerobench directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
tmax = 60

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

# Function to run the JSBSim-based simulation
def run_jsbsim_simulation():
    IC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_cond.xml")
    ROOT_DIR = os.path.dirname(jsbsim.__file__)

    # Initial config
    aircraft_name = "c172p"
    sim_freq_hz = 200

    sim = FGFDMExec(root_dir=ROOT_DIR)
    sim.load_ic(IC_PATH, False)
    sim.load_model(aircraft_name)
    sim.set_dt(1 / sim_freq_hz)

    success = sim.run_ic()
    if not success:
        raise RuntimeError("JSBSim failed to init simulation conditions.")

    data = []
    comp_time = []
    runtime = 0

    while runtime < tmax:
        loop_start_time = time.perf_counter_ns()
        sim["fcs/throttle-cmd-norm"] = float(0.2)
        sim.run()
        outputs = [
            sim["aero/alpha-deg"],
            sim["aero/beta-deg"],
            sim["velocities/vg-fps"] * FT_TO_M,
            sim["velocities/q-rad_sec"] * 180 / math.pi,
            sim["velocities/p-rad_sec"] * 180 / math.pi,
            sim["velocities/r-rad_sec"] * 180 / math.pi

        ]
        data.append(outputs)

        runtime += 1/sim_freq_hz
        fps_recorder = (time.perf_counter_ns() - loop_start_time)
        comp_time.append(fps_recorder)

    return np.arange(0, tmax, 1/200), data, comp_time



# Function to run the second simulation
def run_pyFDM_simulation():
        ### Initial Conditions ###
    power = 5  # engine power level (0-10)
    # Default alpha & beta
    alpha = deg2rad(0)  # Trim Angle of Attack (rad)
    beta = 0  # Side slip angle (rad)

    # Initial Attitude
    alt = 19685  # altitude (ft)
    vt = 500  # initial velocity (ft/sec)

    phi = 0  # Roll angle from wings level (rad)
    theta = 0  # Pitch angle from nose level (rad)
    psi = 0  # Yaw angle from North (rad)


    # Formation clearances according to the formation type
    clearances = [0, 150]  # east and north clearances respectively

    # Define simulation parameters
    step = 1 / 200  # step time
    runtime = 0

    waypoints = []
    # Initial leader position
    leader_position = [[5000, 5000, alt]]  # east, north and altitude
    target_position = [[leader_position[0][0] + clearances[0], leader_position[0][1] + clearances[1], alt]]
    waypoints.append(copy.deepcopy(target_position))

    ap = WaypointAutopilot(target_position, stdout=True)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initial_state_f = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]


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
    integrator = integrator_class(der_func, times[-1], states[-1], **kwargs)

    # Initialize lists for PID data
    comp_time = []
    while runtime < tmax:
        loop_start_time = time.perf_counter_ns()
        # FDM update and states of the follower
        integrator.step()
        dense_output = integrator.dense_output()
        t = times[-1] + step
        times.append(t)
        states.append(dense_output(t))

        runtime += step
        fps_recorder = (time.perf_counter_ns() - loop_start_time)
        comp_time.append(fps_recorder)
    theta = [state[4]  * 180 / math.pi for state in states]
    return np.arange(0, tmax, 1/200), theta[:-1], comp_time

# Run the first simulation
time_jsbsim, data_jsbsim, comp_time_jsbsim = run_jsbsim_simulation()
# Convert data_jsbsim to a NumPy array for easier column access
data_jsbsim = np.array(data_jsbsim)
# Save the array
np.save('data_jsbsim_500v_6000h.npy', data_jsbsim)
# Run the second simulation
#time_pyFDM, theta_pyFDM, comp_time_pyFDM = run_pyFDM_simulation()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

data = pd.read_csv(r'C:\Users\ErenErtugrul\Desktop\Formation\github_repos\AeroBenchVVPython\code\aerobench\examples\waypoint\results.csv')
# Plot the results
fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# Plot Alpha comparison
axs[0, 0].plot(time_jsbsim, data_jsbsim[:, 0], label='JSBSim Alpha', color='green')
axs[0, 0].plot(data['sim_time'], data['alpha'] * 180 / math.pi, label='RustFDM Alpha', color='blue')
axs[0, 0].set_title('Alpha Comparison')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Alpha (deg)')
axs[0, 0].legend()

# Plot Beta comparison
axs[0, 1].plot(time_jsbsim, data_jsbsim[:, 1], label='JSBSim Beta', color='green')
axs[0, 1].plot(data['sim_time'], data['beta'] * 180 / math.pi, label='RustFDM Beta', color='blue')
axs[0, 1].set_title('Beta Comparison')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Beta (deg)')
axs[0, 1].legend()

# Plot Airspeed comparison
axs[1, 0].plot(time_jsbsim, data_jsbsim[:, 2], label='JSBSim Airspeed', color='green')
axs[1, 0].plot(data['sim_time'], data['airspeed'], label='RustFDM Airspeed', color='blue')
axs[1, 0].set_title('Airspeed Comparison')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Airspeed (m/s)')
axs[1, 0].legend()

# Plot Pitch Rate (q) comparison
axs[1, 1].plot(time_jsbsim, data_jsbsim[:, 3], label='JSBSim Pitch Rate (q)', color='green')
axs[1, 1].plot(data['sim_time'], data['q'] * 180 / math.pi, label='RustFDM Pitch Rate (q)', color='blue')
axs[1, 1].set_title('Pitch Rate (q) Comparison')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('q (deg/s)')
axs[1, 1].legend()

# Plot Roll Rate (p) comparison
axs[2, 0].plot(time_jsbsim, data_jsbsim[:, 4], label='JSBSim Roll Rate (p)', color='green')
axs[2, 0].plot(data['sim_time'], data['p'] * 180 / math.pi, label='RustFDM Roll Rate (p)', color='blue')
axs[2, 0].set_title('Roll Rate (p) Comparison')
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].set_ylabel('p (deg/s)')
axs[2, 0].legend()

# Plot Yaw Rate (r) comparison
axs[2, 1].plot(time_jsbsim, data_jsbsim[:, 5], label='JSBSim Yaw Rate (r)', color='green')
axs[2, 1].plot(data['sim_time'], data['r'] * 180 / math.pi, label='RustFDM Yaw Rate (r)', color='blue')
axs[2, 1].set_title('Yaw Rate (r) Comparison')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].set_ylabel('r (deg/s)')
axs[2, 1].legend()

plt.tight_layout()
plt.show()
