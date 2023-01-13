import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from matplotlib_elements import *


N = 100
l = 1
g = 100

# y[2*k+0] = angular position
# y[2*k+1] = angular velocity
def f(t, y):
    y_dot = [None] * (N*2)
    for k in range(N):
        y_dot[2*k] = y[2*k+1]
        y_dot[2*k+1] = -l*(k+1)/g * np.sin(y[2*k])
    return y_dot

t_span = [0, 60]
dt = 0.1
y_init = [np.pi/180*30, 0] * N

sol = solve_ivp(f, t_span, y_init, t_eval=np.arange(t_span[0], t_span[1]+dt, dt))


fig, ax = plt.subplots(figsize=(19.20,10.80), dpi=100)
ax.axis('equal')
ax.axis('off')
fig.tight_layout()

ax.set_xlim(-l*N*1.02, l*N*1.02)
ax.set_ylim(-l*N*1.02, l*N*0.02)

pendulums = []
colors = cm.rainbow(np.linspace(0, 1, N))
for k in range(N):
    pendulum = BallStick(ax, 5, colors[k], 0.05, 'k')
    pendulums.append(pendulum)    

def update(i):
    for k in range(N):
        theta = sol.y[2*k][i]
        x, y = l*(k+1)*np.sin(theta), -l*(k+1)*np.cos(theta)
        pendulums[k].set_ends([0,0], [x,y])
    return pendulums

ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=int(dt*1000)) 

#ani.save('pendulum_wave.mp4', dpi=200, progress_callback=lambda i, n: print(f'\r{i+1}/{n}', end=''))
plt.show()
