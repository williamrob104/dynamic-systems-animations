from collections import deque

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from matplotlib_elements import *

N = 100 # number of pendulums
l = 0.03 # half length of one single pendulum
kappa = l / np.sqrt(3) # radius of gryation
g = 9.8


coef = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if j > i:
            coef[i,j] = (N-j)*4-2
        elif j == i:
            coef[i,j] = (N-j-1)*4
        else:
            coef[i,j] = (N-i)*4-2


# y = [theta1, theta2, ..., theta1_dot, theta2_dot, ...]
def f(t, y):
    theta = np.array(y[0:N])
    theta_dot = np.array(y[N:2*N])

    theta_diff = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            theta_diff[i,j] = theta[j] - theta[i]

    A = coef*np.cos(theta_diff) + np.identity(N) * (1 + (kappa/l)**2)

    b = np.dot(coef*np.sin(theta_diff), theta_dot**2) - g/l * (np.linspace(2*N-1, 1, N) * np.sin(theta))

    theta_ddot = np.linalg.solve(A, b)
    return np.concatenate((theta_dot, theta_ddot))


t_span = [0, 30]
dt = 1./60
y_init = np.concatenate((  np.linspace(20, 120, N) /180*np.pi,   np.zeros(N) ))

def progress(t,y):
    print(f'\rt={t:.3f}', end='')
    return 0


sol = solve_ivp(f, t_span, y_init, t_eval=np.arange(t_span[0], t_span[1]+dt, dt), method='RK45', events=progress)
print()
with open('N_pendulum_sim.npy', 'wb') as f:
    np.save(f, sol.t)
    np.save(f, sol.y)


fig, ax = plt.subplots(figsize=(19.20,10.80), dpi=100)
ax.axis('equal')
ax.axis('off')
fig.tight_layout()

ax.set_xlim(-l*N*1.02, l*N*1.02)
ax.set_ylim(-l*N*1.02, l*N*0.02)

pendulums = []
colors = cm.rainbow(np.linspace(0, 1, N))
for k in range(N):
    pendulum = Stick(ax, 5, colors[k])
    pendulums.append(pendulum)  

path_x = deque()
path_y = deque()
path, = ax.plot([], [], 'k')  

def update(i):
    p = np.array([0.,0.])
    for k in range(N):
        theta = sol.y[k][i]
        v = np.array([l*np.sin(theta), -l*np.cos(theta)])
        pendulums[k].set_ends(p, p+v)
        p += v
    
    path_x.append(p[0])
    path_y.append(p[1])
    if len(path_x) > int(2.0/dt):
        path_x.popleft()
        path_y.popleft()
    path.set_data(path_x, path_y)

    return pendulums, path

ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=int(dt*1000), repeat=False) 

ani.save('N_pendulum.mp4', dpi=200, progress_callback=lambda i, n: print(f'\r{i+1}/{n}', end=''))
#plt.show()
