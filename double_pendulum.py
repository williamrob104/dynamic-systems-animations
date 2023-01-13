from collections import deque
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp

from matplotlib_elements import *


g = 9.81
m1 = 1 # mass of pendulum 1
l1 = 0.5 # half the length of pendulum 1
m2 = 1 # mass of pendulum 2
l2 = 0.5 # half the length of pendulum 2

I1 = m1 * (2*l1)**2 / 12 # moment of inertia of pendulum 1
I2 = m2 * (2*l2)**2 / 12 # moment of inertia of pendulum 2


# y = [theta1, theta1_dot, theta2, theta2_dot]
def y_dot_func(t, y):
    theta1, theta1_dot, theta2, theta2_dot = y 

    sin_diff = np.sin(theta1 - theta2)
    cos_diff = np.cos(theta1 - theta2)

    a11 = 2*m2*l1*l2*cos_diff
    a12 = I2 + m2*l2*l2
    b1 = 2*m2*l1*l2*theta1_dot**2*sin_diff - m2*g*l2*np.sin(theta2)

    a21 = I1 + m1*l1*l1 + 4*m2*l1*l1 
    a22 = 2*m2*l1*l2*cos_diff
    b2 = -m1*g*l1*np.sin(theta1) - 2*m2*l1*l2*theta2_dot**2*sin_diff - 2*m2*g*l1*np.sin(theta1)

    theta1_dot_dot, theta2_dot_dot = np.linalg.solve([[a11,a12],[a21,a22]], [b1, b2])

    return [theta1_dot, theta1_dot_dot, theta2_dot, theta2_dot_dot]



t_span = [0, 60]
dt = 0.1
y_init = [np.pi/180*30, 1, np.pi/180*80, -5]

sol = solve_ivp(y_dot_func, t_span, y_init, t_eval=np.arange(t_span[0], t_span[1]+dt, dt))

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[2,1])

ax1 = fig.add_subplot(spec[:, 0])
ax1.axis('equal')
ax1.set_axis_off()
ax1.set_xlim(-2*(l1+l2)*1.2, 2*(l1+l2)*1.2)
ax1.set_ylim(-2*(l1+l2)*1.2, 2*(l1+l2)*0.3)

ax2 = fig.add_subplot(spec[0, -1])
ax2.set_xlabel(r'$\theta_1$ (rad)')
ax2.set_ylabel(r'$\dot{\theta_1}$ (rad/s)')

ax3 = fig.add_subplot(spec[1, -1])
ax3.set_xlabel(r'$\theta_2$ (rad)')
ax3.set_ylabel(r'$\dot{\theta_2}$ (rad/s)')

fig.tight_layout()

pendulum1 = Stick(ax1, 5, 'b')
pendulum2 = Stick(ax1, 5, 'r')
path_plot,= ax1.plot([], [], 'k')
path = (deque(), deque())

phase1_plot, = ax2.plot([], [], 'b')
phase2_plot, = ax3.plot([], [], 'r')
phase1 = ([], [])
phase2 = ([], [])
    

def update(i):
    theta1, theta1_dot, theta2, theta2_dot = sol.y[0][i], sol.y[1][i], sol.y[2][i], sol.y[3][i]
    t = sol.t[i]

    p1 = np.array([2*l1*np.sin(theta1), -2*l1*np.cos(theta1)])
    p2 = p1 + np.array([2*l2*np.sin(theta2), -2*l2*np.cos(theta2)])

    pendulum1.set_ends(np.array([0,0]), p1)
    pendulum2.set_ends(p1, p2)

    path[0].append(p2[0])
    path[1].append(p2[1])
    if len(path[0]) > int(2.0 / dt):
        path[0].popleft()
        path[1].popleft()
    path_plot.set_data(path[0], path[1])

    phase1[0].append(theta1)
    phase1[1].append(theta1_dot)
    phase1_plot.set_data(phase1[0], phase1[1])
    ax2.set_xlim(np.min(phase1[0])-0.2, np.max(phase1[0])+0.2)
    ax2.set_ylim(np.min(phase1[1])-0.2, np.max(phase1[1])+0.2)

    phase2[0].append(theta2)
    phase2[1].append(theta2_dot)
    phase2_plot.set_data(phase2[0], phase2[1])
    ax3.set_xlim(np.min(phase2[0])-0.2, np.max(phase2[0])+0.2)
    ax3.set_ylim(np.min(phase2[1])-0.2, np.max(phase2[1])+0.2)

    return pendulum1, pendulum2, phase1_plot, phase2_plot



ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=int(dt*1000), repeat=False) 

#ani.save('double_pendulum_2.mp4', dpi=300, progress_callback=lambda i, n: print(f'\r{i+1}/{n}', end=''))
plt.show()
