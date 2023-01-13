import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp

from matplotlib_elements import *


g = 9.81
k = 1 # spring constant
m1 = 1 # mass of ball 1
l1 = 1 # length of pendulum 1
r1 = 0.8 # connection length of spring on pendulum 1
m2 = 1 # mass of ball 1
l2 = 1 # length of pendulum 1
r2 = 0.8 # connection length of spring on pendulum 1
d = 0.5 # distance between pendulums and free length of spring

# y = [theta1, theta1_dot, theta2, theta2_dot]
def y_dot_func(t, y):
    theta1, theta1_dot, theta2, theta2_dot = y
    p1 = np.array([r1*np.sin(theta1), -r1*np.cos(theta1)])
    p2 = np.array([r2*np.sin(theta2), -r1*np.cos(theta2)])
    fs1 = np.array([d, 0]) + p2 - p1
    fs1 = k * (1 - d/norm(fs1)) * fs1
    fs2 = -fs1 
    ts1 = np.cross(p1, fs1)
    ts2 = np.cross(p2, fs2)

    theta1_dot_dot = ts1/(m1*l1*l1) - g/l1*np.sin(theta1)
    theta2_dot_dot = ts2/(m2*l2*l2) - g/l2*np.sin(theta2)
    return np.array([theta1_dot, theta1_dot_dot, theta2_dot, theta2_dot_dot])

t_span = [0, 60]
dt = 0.1
y_init = [-np.pi/180*0, 0, np.pi/180*50, 0]

sol = solve_ivp(y_dot_func, t_span, y_init, t_eval=np.arange(t_span[0], t_span[1]+dt, dt))

fig = plt.figure(figsize=(12, 6), dpi=100)
spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1.8,1])

ax1 = fig.add_subplot(spec[:, 0])
ax1.axis('equal')
ax1.set_axis_off()
ax1.set_xlim(-(d/2+l1)*1.2, (d/2+l2)*1.2)
ax1.set_ylim(-max(l1,l2)*1.2, max(l1,l2)*0.2)

ax3 = fig.add_subplot(spec[1, -1])
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 0.1)
ax3.set_xlabel('t (s)')
ax3.set_ylabel(r'$energy$ (J)')
fig.tight_layout()

ax2 = fig.add_subplot(spec[0, -1], sharex=ax3)
ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax2.set_ylim(-np.pi/2, np.pi/2)
ax2.set_ylabel(r'$\theta$ (rad)')

pendulum1 = BallStick(ax1, 10, 'b', 1, 'k')
pendulum2 = BallStick(ax1, 10, 'r', 1, 'k')
spring = Spring(ax1, d, 6, linewidth=1, color='g')

theta1_plot, theta2_plot = ax2.plot([], [], 'b', [], [], 'r')
theta1_arr = []
theta2_arr = []
t_arr = []

e_k_arr, e_h1_arr, e_v1_arr, e_h2_arr, e_v2_arr, e_total_arr = [], [], [], [], [], []
e_k_plot, = ax3.plot([], [], 'g', label=r'$U_{spring}$')
e_h1_plot, = ax3.plot([], [], 'b', label=r'$U_1$')
e_v1_plot, = ax3.plot([], [], 'm', label=r'$K_1$')
e_h2_plot, = ax3.plot([], [], 'r', label=r'$U_2$')
e_v2_plot, = ax3.plot([], [], 'y', label=r'$K_2$')
e_total_plot, = ax3.plot([], [], 'k', label=r'$E_{total}$')
ax3.legend(loc='lower left')
    

def update(i):
    theta1, theta1_dot, theta2, theta2_dot = sol.y[0][i], sol.y[1][i], sol.y[2][i], sol.y[3][i]
    t = sol.t[i]

    x1, y1 = -d/2 + l1*np.sin(theta1), -l1*np.cos(theta1)
    x2, y2 = d/2 + l2*np.sin(theta2), -l2*np.cos(theta2)
    p1 = np.array([-d/2 + r1*np.sin(theta1), -r1*np.cos(theta1)])
    p2 = np.array([d/2 + r2*np.sin(theta2), -r2*np.cos(theta2)])
    pendulum1.set_ends([-d/2,0], [x1,y1])
    pendulum2.set_ends([d/2,0], [x2,y2])
    spring.set_ends(p1, p2)

    theta1_arr.append(theta1)
    theta2_arr.append(theta2)
    t_arr.append(t)
    theta1_plot.set_data(t_arr, theta1_arr)
    theta2_plot.set_data(t_arr, theta2_arr)

    e_k_arr.append(0.5 * k * (d - norm(p2-p1))**2)
    e_h1_arr.append(m1 * g * (y1 + l1))
    e_v1_arr.append(0.5 * m1 * (l1*theta1_dot)**2)
    e_h2_arr.append(m2 * g *(y2 + l2))
    e_v2_arr.append(0.5 * m2 * (l2*theta2_dot)**2)
    e_total_arr.append(e_k_arr[-1] + e_h1_arr[-1] + e_v1_arr[-1] + e_h2_arr[-1] + e_v2_arr[-1])
    e_k_plot.set_data(t_arr, e_k_arr)
    e_h1_plot.set_data(t_arr, e_h1_arr)
    e_v1_plot.set_data(t_arr, e_v1_arr)
    e_h2_plot.set_data(t_arr, e_h2_arr)
    e_v2_plot.set_data(t_arr, e_v2_arr)
    e_total_plot.set_data(t_arr, e_total_arr)
    if e_total_arr[-1] > ax3.get_ylim()[1]:
        ax3.set_ylim(0, e_total_arr[-1])
    if t > 1:
        ax2.set_xlim(0, t)

    return pendulum1, pendulum2, spring, theta1_plot, theta2_plot, e_k_plot, e_h1_plot, e_v1_plot, e_h2_plot, e_v2_plot, e_total_plot



ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=int(dt*1000), repeat=False) 

#ani.save('coupled_pendulum.mp4', dpi=320, progress_callback=lambda i, n: print(f'\r{i+1}/{n}', end=''))
plt.show()
