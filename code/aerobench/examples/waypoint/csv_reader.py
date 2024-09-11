import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

data = pd.read_csv(r'C:\Users\ErenErtugrul\Desktop\Formation\github_repos\AeroBenchVVPython\code\aerobench\examples\waypoint\outputs_pau_fdm.csv')
data2 = pd.read_csv(r'C:\Users\ErenErtugrul\Desktop\Formation\github_repos\AeroBenchVVPython\code\aerobench\examples\waypoint\results_jsb.txt', delimiter=",", header=None, names=['time', 'computation_time', 'altitude', 'airspeed', 'alpha', 'theta'])
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8))

axs2[0].plot(data['sim_time'], data['theta'], label='Pau FDM Theta', color='green')
axs2[0].plot(data2['time'], data2['theta'], label='JSBSim Theta', color='blue')
axs2[0].set_title('Theta Comparison')
axs2[0].set_xlabel('Time (s)')
axs2[0].set_ylabel('Theta (deg)')
axs2[0].legend()


# Plot computation time comparison
axs2[1].plot(data['sim_time'], data['computation_time'], label='Pau FDM Comp. Time', color='red')
axs2[1].plot(data2['time'], data2['computation_time'], label='JSBSim Comp. Time', color='orange', alpha=0.5)
axs2[1].set_title('Computation Time Comparison')
axs2[1].set_xlabel('Time (s)')
axs2[1].set_ylabel('Comp. Time (ns)')
axs2[1].legend()

plt.show()