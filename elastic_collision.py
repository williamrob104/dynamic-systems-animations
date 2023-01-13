import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from matplotlib_elements import *


m1 = 2
m2 = 1
v1_init = 1
v2_init = 0
k = 10
free_length = 2

l1 = np.sqrt(m1) * 0.5
l2 = np.sqrt(m2) * 0.5

# y = [x1, x2, v1, v2]
def y_dot_func(t, y):
    x1, x2, v1, v2 = y
    a1 = a2 = 0
    if np.abs(x2 - x1) < free_length:
        F = k * (free_length - np.abs(x2 - x1))
        a1 = -F / m1 
        a2 = F / m2
    return v1, v2, a1, a2

t_span = [0, 5]
dt = 0.05
y_init = [0, free_length+1, v1_init, v2_init]

sol = solve_ivp(y_dot_func, t_span, y_init, t_eval=np.arange(t_span[0], t_span[1]+dt, dt))

# ----- plot -----
fig = plt.figure(figsize=(12, 6), dpi=100)
spec = fig.add_gridspec(ncols=2, nrows=3, width_ratios=[1.8,1])

# simulation
ax1 = fig.add_subplot(spec[:, 0])
ax1.axis('equal')
ax1.set_axis_off()
ax1.set_xlim(-1, 5)
ax1.set_ylim(-3, 4)

block1 = patches.Rectangle((0.0, 0.0), l1,l1, facecolor='r')
ax1.add_patch(block1)

block2 = patches.Rectangle((0.0, 0.0), l2,l2, facecolor='b')
ax1.add_patch(block2)

spring = Spring(ax1, free_length, color='m')

# energy plot
ax4 = fig.add_subplot(spec[2, -1])
ax4.set_xlim(0, 0.5)
ax4.set_ylim(0, 0.5*m1*v1_init**2+0.5*m2*v2_init**2)
ax4.set_ylim(ax4.get_ylim()[0]-(ax4.get_ylim()[1]-ax4.get_ylim()[0])*0.05, ax4.get_ylim()[1]+(ax4.get_ylim()[1]-ax4.get_ylim()[0])*0.05)
ax4.set_ylabel(r'energy (J)')
K1_plot, = ax4.plot([], [], 'r', label='$K_1$')
K2_plot, = ax4.plot([], [], 'b', label='$K_2$')
U_plot, = ax4.plot([], [], 'm', label=r'$U_{spring}$')
E_plot, = ax4.plot([], [], 'y--', label='$E$')
K1_data, K2_data, U_data, E_data, t_data = [], [], [], [], []

# velocity plot
ax2 = fig.add_subplot(spec[0, -1], sharex=ax4)
ax2.set_ylim(min(np.min(sol.y[2]), np.min(sol.y[3])), max(np.max(sol.y[2]), np.max(sol.y[3])))
ax2.set_ylim(ax2.get_ylim()[0]-(ax2.get_ylim()[1]-ax2.get_ylim()[0])*0.05, ax2.get_ylim()[1]+(ax2.get_ylim()[1]-ax2.get_ylim()[0])*0.05)
ax2.set_ylabel(r'$v$ (m/s)')
v1_plot, = ax2.plot([], [], 'r', label=r'$v_1$')
v2_plot, = ax2.plot([], [], 'b', label=r'$v_2$')
vc_plot, = ax2.plot([], [], 'y--', label=r'$v_c$')

# momentum plot
ax3 = fig.add_subplot(spec[1, -1], sharex=ax4)
ax3.set_ylim(min(np.min(m1*sol.y[2]), np.min(m2*sol.y[3])), max(np.max(m1*sol.y[2]), np.max(m2*sol.y[3])))
ax3.set_ylim(ax3.get_ylim()[0]-(ax3.get_ylim()[1]-ax3.get_ylim()[0])*0.1, ax3.get_ylim()[1]+(ax3.get_ylim()[1]-ax3.get_ylim()[0])*0.1)
ax3.set_xlabel('t (s)')
ax3.set_ylabel(r'momentum (kg$\cdot$m/s)')
p1_plot, = ax3.plot([], [], 'r', label=r'$p_1$')
p2_plot, = ax3.plot([], [], 'b', label=r'$p_2$')
pc_plot, = ax3.plot([], [], 'y--', label=r'$p_c$')

fig.tight_layout()

    

def update(i):
    x1, x2, v1, v2 = sol.y[0][i], sol.y[1][i], sol.y[2][i], sol.y[3][i]
    t = sol.t[i]

    block1.set_x(x1-l1)
    block2.set_x(x2)

    spring_length = free_length
    if np.abs(x2 - x1) < free_length:
        spring_length = np.abs(x2 - x1)
    spring.set_ends([x2-spring_length, min(l1,l2)/2], [x2, min(l1,l2)/2])

    v1_plot.set_data(sol.t[:i+1], sol.y[2][:i+1])
    v2_plot.set_data(sol.t[:i+1], sol.y[3][:i+1])
    vc_plot.set_data(sol.t[:i+1], (sol.y[2][:i+1]*m1 + sol.y[3][:i+1]*m2)/(m1+m2))
    ax2.legend(loc='upper right')

    p1_plot.set_data(sol.t[:i+1], m1*sol.y[2][:i+1])
    p2_plot.set_data(sol.t[:i+1], m2*sol.y[3][:i+1])
    pc_plot.set_data(sol.t[:i+1], m1*sol.y[2][:i+1]+m2*sol.y[3][:i+1])
    ax3.legend(loc='upper right')

    K1_data.append(0.5*m1*v1**2)
    K2_data.append(0.5*m2*v2**2)
    U_data.append(0.5*k*(free_length - np.abs(x2-x1))**2 if np.abs(x2-x1) < free_length else 0)
    E_data.append(K1_data[-1] + K2_data[-1] + U_data[-1])
    t_data.append(t)
    K1_plot.set_data(t_data, K1_data)
    K2_plot.set_data(t_data, K2_data)
    U_plot.set_data(t_data, U_data)
    E_plot.set_data(t_data, E_data)
    ax4.legend(loc='upper right')

    if t > 0.5:
        ax4.set_xlim(0, t)

ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=int(dt*1000), repeat=False) 

#ani.save('coupled_pendulum.mp4', dpi=320, progress_callback=lambda i, n: print(f'\r{i+1}/{n}', end=''))
plt.show()
