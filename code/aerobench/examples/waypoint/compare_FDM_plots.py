import os  # noqa: I001
import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_jsbsim = np.load('data_jsbsim_500v_6000h.npy')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

data = pd.read_csv(r'C:\Users\ErenErtugrul\Desktop\Formation\github_repos\AeroBenchVVPython\code\aerobench\examples\waypoint\results.csv')
# Plot the results
fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# Plot Alpha comparison
axs[0, 0].plot(data['sim_time'], data_jsbsim[:, 0], label='JSBSim Alpha', color='green')
axs[0, 0].plot(data['sim_time'], data['alpha'] * 180 / math.pi, label='RustFDM Alpha', color='blue')
axs[0, 0].set_title('Alpha Comparison')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Alpha (deg)')
axs[0, 0].legend()

# Plot Beta comparison
axs[0, 1].plot(data['sim_time'], data_jsbsim[:, 1], label='JSBSim Beta', color='green')
axs[0, 1].plot(data['sim_time'], data['beta'] * 180 / math.pi, label='RustFDM Beta', color='blue')
axs[0, 1].set_title('Beta Comparison')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Beta (deg)')
axs[0, 1].legend()

# Plot Airspeed comparison
axs[1, 0].plot(data['sim_time'], data_jsbsim[:, 2], label='JSBSim Airspeed', color='green')
axs[1, 0].plot(data['sim_time'], data['airspeed'], label='RustFDM Airspeed', color='blue')
axs[1, 0].set_title('Airspeed Comparison')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Airspeed (m/s)')
axs[1, 0].legend()

# Plot Pitch Rate (q) comparison
axs[1, 1].plot(data['sim_time'], data_jsbsim[:, 3], label='JSBSim Pitch Rate (q)', color='green')
axs[1, 1].plot(data['sim_time'], data['q'] * 180 / math.pi, label='RustFDM Pitch Rate (q)', color='blue')
axs[1, 1].set_title('Pitch Rate (q) Comparison')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('q (deg/s)')
axs[1, 1].legend()

# Plot Roll Rate (p) comparison
axs[2, 0].plot(data['sim_time'], data_jsbsim[:, 4], label='JSBSim Roll Rate (p)', color='green')
axs[2, 0].plot(data['sim_time'], data['p'] * 180 / math.pi, label='RustFDM Roll Rate (p)', color='blue')
axs[2, 0].set_title('Roll Rate (p) Comparison')
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].set_ylabel('p (deg/s)')
axs[2, 0].legend()

# Plot Yaw Rate (r) comparison
axs[2, 1].plot(data['sim_time'], data_jsbsim[:, 5], label='JSBSim Yaw Rate (r)', color='green')
axs[2, 1].plot(data['sim_time'], data['r'] * 180 / math.pi, label='RustFDM Yaw Rate (r)', color='blue')
axs[2, 1].set_title('Yaw Rate (r) Comparison')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].set_ylabel('r (deg/s)')
axs[2, 1].legend()

plt.tight_layout()
plt.show()
