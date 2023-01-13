import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from scipy.integrate import solve_ivp

from matplotlib_elements import *

N = 3 # number of masses
m = np.ones(N)*0.1 # mass of masses
k = 2 # stiffness of spring

l = 0.5 # free length of spring
a = l * 0.6 # side length of mass

M = np.diag(m) # mass matrix
K = np.zeros((N,N)) # spring matrix
for i in range(N):
    K[i,i] = 2
    if i > 0:
        K[i, i-1] = -1
    if i < N-1:
        K[i, i+1] = -1
K *= k


# y = [x1 x2 ... xN x1_dot x2_dot ... xN_dot]
def f(t, y):
    x = y[0:N]
    x_dot = y[N:2*N]

    x_ddot = np.linalg.solve(M, -(K@x))

    return np.concatenate((x_dot, x_ddot))



t_span = [0, 60]
dt = 1./15

y_init = np.concatenate((  (np.random.rand(N)*2-1)*0.2,  np.zeros(N) ))

def progress(t,y):
    print(f'\rt={t:.3f}', end='')
    return 0

sol = solve_ivp(f, t_span, y_init, t_eval=np.arange(t_span[0], t_span[1], dt), method='RK45', events=progress)
print()


fig = plt.figure(figsize=(9.6,3.6), dpi=100)
gs = fig.add_gridspec(2,2, width_ratios=[3,1])
ax = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

ax.axis('equal')
ax.axis('off')

ax.set_xlim(0, l*(N+1)+a*N)

colors = cm.rainbow(np.linspace(0, 1, N))
masses = []
for i in range(N):
    mass = Rectangle(ax, a, a, colors[i])
    masses.append(mass)

springs = []
for i in range(N+1):
    spring = Spring(ax, l, color='k')
    springs.append(spring)

ax.plot((0,0), (-a,a), 'k')
ax.plot((l*(N+1)+a*N,l*(N+1)+a*N), (-a,a), 'k')


ax2.set_xlim(0,10)
ax2.set_ylim(np.min(np.min(sol.y[0:N])), np.max(np.max(sol.y[0:N])))
ax2.set_xlabel('$t$ (s)')
ax2.set_ylabel('mass offset')

x_plots = []
for i in range(N):
    x_plot, = ax2.plot([], [], color=colors[i])
    x_plots.append(x_plot)

ax3.set_xlabel('frequency (Hz)')
ax3.set_xlim(0,2)

for i in range(N):
    xi = np.pad(sol.y[i], (0, 0))
    freq = fft.rfftfreq(len(xi), dt)
    fft_xi = fft.rfft(xi)
    ax3.plot(freq, 2.0*dt * np.abs(fft_xi), color=colors[i])

    
fig.tight_layout()


def update(frame):
    prev_center = -a*0.5
    for i in range(N):
        x = sol.y[i][frame]
        x_dot = sol.y[N+i][frame]

        curr_center = l*(i+1)+a*i+a*0.5+x
        masses[i].set_center(curr_center, 0)
        springs[i].set_ends((curr_center-a*0.5, 0), (prev_center+a*0.5, 0))
        prev_center = curr_center
    springs[N].set_ends((prev_center+a*0.5, 0), (l*(N+1)+a*N, 0))

    for i in range(N):
        x_plots[i].set_data(sol.t[0:frame+1], sol.y[i][0:frame+1])
    if sol.t[frame] > 10:
        ax2.set_xlim(sol.t[frame]-10, sol.t[frame])


ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=int(dt*1000), repeat=False) 

#ani.save('free_mass_spring.mp4', dpi=400, progress_callback=lambda i, n: print(f'\r{i+1}/{n}', end=''))
plt.show()
