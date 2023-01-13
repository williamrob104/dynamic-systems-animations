import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from matplotlib_elements import *

N = 5 # number of pendulums
l = np.ones(N)*0.2 # length of pendulums
r = l * 0.3 # distance between pendulum center of mass ans pivot
m = np.ones(N)*1 # mass of pendulum
M = 10 # mass of cart
espilon = 1 # escapement coefficient
theta_0 = np.radians(5) # min theta
g = 9.8


I_pivot = m*l**2/12 + m*r**2 # moment of inertial about pivot
d = np.max(l) * 0.3 # distance between pendulums


# y = [theta1, theta2, ..., theta1_dot, theta2_dot, ..., x, x_dot]
def f(t, y):
    theta = np.array(y[0:N])
    theta_dot = np.array(y[N:2*N])
    x = y[-2]
    x_dot = y[-1]

    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
    for k in range(N):
        A[k, k] = 1
        A[k, -1] = m[k]*r[k]*np.cos(theta[k])/I_pivot[k]
        b[k] = -m[k]*r[k]*g/I_pivot[k] * np.sin(theta[k]) - espilon*((theta[k]/theta_0)**2-1)*theta_dot[k]

        A[-1, k] = m[k]*r[k]*np.cos(theta[k])
    A[-1, -1] = M + np.sum(m)
    b[-1] = np.sum(m*r*theta_dot**2*np.sin(theta))
    ans = np.linalg.solve(A,b)

    theta_ddot = ans[0:-1]
    x_ddot = ans[-1]

    return np.concatenate((  theta_dot, theta_ddot, np.array([x_dot, x_ddot])  ))


t_span = [0, 120]
dt = 1./30

theta_init = (np.random.rand(N)*np.radians(20)+theta_0) * np.sign(np.random.rand(N)-0.5)
y_init = np.concatenate((  theta_init,  np.zeros(N), np.array([0, 0])  ))

def progress(t,y):
    print(f'\rt={t:.3f}', end='')
    return 0


sol = solve_ivp(f, t_span, y_init, t_eval=np.arange(t_span[0], t_span[1], dt), method='RK45', events=progress)
print()

fig = plt.figure(figsize=(6.4,3.6), dpi=100)
gs = fig.add_gridspec(2,2, width_ratios=[3,1])
ax = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

ax.axis('equal')
ax.axis('off')

ax2.axis('equal')
ax2.axis('off')

ax3.set_xlabel('$t$')
ax3.set_ylabel('$r$')
ax3.set_ylim(0, 1.1)
ax3.set_xlim(0, 1)

fig.tight_layout()

ax.set_xlim(-d*2, d*N+d)
ax.set_ylim(-np.max(l)-0.5, 0.5)

colors = cm.rainbow(np.linspace(0, 1, N))
pendulums = []
for k in range(N):
    pendulum = Stick(ax, 5, colors[k])
    pendulums.append(pendulum)
cart = Stick(ax, 15, [0.5,0.5,0.5, 0.3]) 

temp = np.linspace(0, 2*np.pi)
ax2.plot(np.cos(temp), np.sin(temp), 'k')
phase_plot = ax2.scatter(np.zeros(N), np.zeros(N), c=colors)

mag_plot, = ax3.plot([], [], 'k')
mag_data, t_data = [], []

def update(i):
    x = sol.y[-2][i]
    x_dot = sol.y[-1][i]
    t = sol.t[i]

    x_momentum = M*x_dot
    phases = []
    for k in range(N):
        theta = sol.y[k][i]
        theta_dot = sol.y[N+k][i]

        p1 = np.array([x + d*k, 0])
        p2 = p1 + r[k]*np.array([-np.sin(theta), np.cos(theta)])
        p3 = p2 + l[k]*np.array([np.sin(theta), -np.cos(theta)])

        phi_dot = np.sqrt(g/l[k])
        phi = np.arctan2(-theta_dot/phi_dot, theta)
        phases.append(phi)

        pendulums[k].set_ends(p2, p3)

        x_momentum += m[k] * (x_dot + l[k]*theta_dot*np.cos(theta))

    cart.set_ends([x, 0], [x+d*(N-1), 0])

    phases = np.array(phases)
    sin_phases = np.sin(phases)
    cos_phases = np.cos(phases)
    phase_plot.set_offsets(np.stack([cos_phases, sin_phases], axis=1))
    
    mag = np.hypot(np.mean(sin_phases), np.mean(cos_phases))
    mag_data.append(mag)
    t_data.append(t)
    mag_plot.set_data(t_data, mag_data)
    if t > 1:
        ax3.set_xlim(0, t)


ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=int(dt*1000), repeat=False) 

#ani.save('sync.mp4', dpi=200, progress_callback=lambda i, n: print(f'\r{i+1}/{n}', end=''))
plt.show()