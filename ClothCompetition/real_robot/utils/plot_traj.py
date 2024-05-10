import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.style'] = 'normal'

traj_log = np.load('../../real_exp/log/traj_log.npy')
traj_desired = np.load('../../real_exp/log/traj_desired.npy')

# plot two subplots: one for move log one for trajectory log

time_log = traj_log[:,0] - traj_log[0,0]
traj_log_x = traj_log[:,1]
traj_log_y = traj_log[:,2]
traj_log_z = traj_log[:,3]

# find the index of start of trajectory, the x will first decrease then increase
diff_signal = np.diff(traj_log_x)
for i in range(len(diff_signal)):
    if abs(diff_signal[i]) > 2e-4:
        index_start = i
        break
steps = traj_desired.shape[0]

traj_log_x_index = traj_log_x[index_start:index_start+steps]
traj_log_y_index = traj_log_y[index_start:index_start+steps]
traj_log_z_index = traj_log_z[index_start:index_start+steps]
time_log_index = time_log[index_start:index_start+steps]

time_desired = traj_desired[:,0] - traj_desired[0,0]
traj_desired_x = traj_desired[:,1]
traj_desired_y = traj_desired[:,2]
traj_desired_z = traj_desired[:,3]

traj_log_x_index = (traj_log_x_index - traj_log_x_index[0]) *(-1)
traj_log_y_index = (traj_log_y_index - traj_log_y_index[0]) *(-1)
traj_log_z_index = traj_log_z_index - traj_log_z_index[-1]

traj_desired_x = (traj_desired_x - traj_desired_x[0]) *(-1)
traj_desired_y = (traj_desired_y - traj_desired_y[0]) *(-1)
traj_desired_z = traj_desired_z - traj_desired_z[-1]


log_x_vel = np.diff(traj_log_x_index)*100
log_y_vel = np.diff(traj_log_y_index)*100
log_z_vel = np.diff(traj_log_z_index)*100

desired_x_vel = np.diff(traj_desired_x)*100
desired_y_vel = np.diff(traj_desired_y)*100
desired_z_vel = np.diff(traj_desired_z)*100

log_x_acc = np.diff(np.diff(traj_log_x_index)*100)*100
log_y_acc = np.diff(np.diff(traj_log_y_index)*100)*100
log_z_acc = np.diff(np.diff(traj_log_z_index)*100)*100

desired_x_acc = np.diff(np.diff(traj_desired_x)*100)*100
desired_y_acc = np.diff(np.diff(traj_desired_y)*100)*100
desired_z_acc = np.diff(np.diff(traj_desired_z)*100)*100

# six subplots - pos and vel for x, y, z
fig, ((ax1, ax2, ax3), (ax7, ax8, ax9), (ax4, ax5, ax6)) = plt.subplots(3, 3)
ax1.plot(traj_log_x_index, label='Move Log')
ax1.plot(traj_desired_x, label='Desired Trajectory')
ax1.set_title('pos x')
ax1.set_ylabel('Traj Log')
ax1.legend()

ax2.plot(traj_log_y_index, label='Move Log')
ax2.plot(traj_desired_y, label='Desired Trajectory')
ax2.set_title('pos y')
ax2.set_ylabel('Traj Log')
ax2.legend()

ax3.plot(traj_log_z_index, label='Move Log')
ax3.plot(traj_desired_z, label='Desired Trajectory')
ax3.set_title('pos z')
ax3.set_ylabel('Traj Log')
ax3.legend()

ax4.plot(log_x_acc, label='Move Log')
ax4.plot(desired_x_acc, label='Desired Trajectory')
ax4.set_title('acc x')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Traj Log')
ax4.legend()

ax5.plot(log_y_acc, label='Move Log')
ax5.plot(desired_y_acc, label='Desired Trajectory')
ax5.set_title('acc y')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Traj Log')
ax5.legend()

ax6.plot(log_z_acc, label='Move Log')
ax6.plot(desired_z_acc, label='Desired Trajectory')
ax6.set_title('acc z')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Traj Log')
ax6.legend()


ax7.plot(log_x_vel, label='Move Log')
ax7.plot(desired_x_vel, label='Desired Trajectory')
ax7.set_title('vel x')
ax7.set_ylabel('Traj Log')
ax7.legend()

ax8.plot(log_y_vel, label='Move Log')
ax8.plot(desired_y_vel, label='Desired Trajectory')
ax8.set_title('vel y')
ax8.set_ylabel('Traj Log')
ax8.legend()

ax9.plot(log_z_vel, label='Move Log')
ax9.plot(desired_z_vel, label='Desired Trajectory')
ax9.set_title('vel z')
ax9.set_ylabel('Traj Log')

# plt.tight_layout()
plt.legend()
plt.show()

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(time_log, traj_log_x)
# ax1.set_title('Traj Log')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Traj Log')

# ax2.plot(time_desired, traj_desired_x)
# ax2.set_title('Desired Trajectory')
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Desired Trajectory')

# plt.tight_layout()
# plt.show()

# ignore time, plot move_log and trajectory_log on same plot
# plt.plot(traj_log_x_index, label='Move Log')
# plt.plot(traj_desired_x, label='Trajectory Log')
# plt.title('Move Log and Trajectory Log')
# plt.xlabel('Time (s)')
# plt.ylabel('Log')
# plt.legend()
# plt.show()

