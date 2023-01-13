import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from matplotlib_elements import *


a = 0.7  # joint half length
b = 0.5  # joint half width
l = 2.5  # bar length
beta = 60/180.0*np.pi # skew angle


fig = plt.figure(figsize=(6.4,4.8), dpi=100)
spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, height_ratios=[2,1])
ax = fig.add_subplot(spec[0,:], projection='3d')

verts = [(0,0,0), (l,0,0), (l+a,b,0), (l,b,0), (l,-b,0), (l+a,-b,0)]
vert_connections = [[0,1], [2,3,4,5], [2,5]]
bar1 = MovableLine3DCollection(ax, verts, vert_connections, colors=['r','r','k'])
bar1_rotation_center = verts[0]
bar1_rotation_axis = (1,0,0)

bar2_1 = MovableLine3DCollection(ax, verts, vert_connections, colors=['b','b','k'])
bar2_1.rotate((l+a,0,0), (0,1,0), np.pi)
bar2_1.rotate((l+a,0,0), (1,0,0), np.pi/2)
temp = bar2_1.rotate((l+a,0,0), (0,0,1), beta)
bar2_rotation_center = temp[0,:]
bar2_rotation_axis = (np.cos(beta), np.sin(beta), 0)
bar2_2 = MovableLine3DCollection(ax, temp, vert_connections, colors=['b','b','k'])
bar2_2.rotate(bar2_rotation_center, (0,0,1), np.pi)

bar3 = MovableLine3DCollection(ax, verts, vert_connections, colors=['y','y','k'])
bar3.rotate((l+a,0,0), (0,1,0), np.pi)
temp = bar3.translate(2*(l+a) * np.array([np.cos(beta), np.sin(beta), 0]))
bar3_rotation_center = temp[0,:]
bar3_rotation_axis = (1,0,0)

ax.set_axis_off()
ax.set_xlim(3, 7)
ax.set_ylim(0, 4)
ax.set_zlim(-4, 0)


ax2 = fig.add_subplot(spec[1,:])
omega1_plot, = ax2.plot([], [], 'r', label='input')
omega2_plot, = ax2.plot([], [], 'b', label='center')
omega3_plot, = ax2.plot([], [], 'y--', label='output')
ax2.legend(loc='upper right')
ax2.set_xlim(0,3)
ax2.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\omega$')

fig.tight_layout()

theta1, theta2, theta3 = 0, 0, np.pi/2
omega1, omega2, omega3, t = [], [], [], []
def update(i):
    global theta1, theta2, theta3

    dt = 0.02
    d_theta1 = 5/180.0*np.pi
    theta1 += d_theta1
    bar1.rotate(bar1_rotation_center, bar1_rotation_axis, d_theta1)

    d_theta2 = np.arctan2(np.sin(theta1), np.cos(beta)*np.cos(theta1)) - theta2
    while d_theta2 < 0:
        d_theta2 += 2*np.pi
    theta2 += d_theta2
    bar2_1.rotate(bar2_rotation_center, bar2_rotation_axis, d_theta2)
    bar2_2.rotate(bar2_rotation_center, bar2_rotation_axis, d_theta2)
    
    d_theta3 = np.arctan2(np.sin(theta2+np.pi/2), np.cos(beta)*np.cos(theta2+np.pi/2)) - theta3
    while d_theta3 < 0:
        d_theta3 += 2*np.pi
    theta3 += d_theta3
    bar3.rotate(bar3_rotation_center, bar3_rotation_axis, d_theta3)

    t.append(0. if len(t)==0 else t[-1]+dt)
    omega1.append(d_theta1/dt)
    omega2.append(d_theta2/dt)
    omega3.append(d_theta3/dt)
    omega1_plot.set_data(t, omega1)
    omega2_plot.set_data(t, omega2)
    omega3_plot.set_data(t, omega3)
    if t[-1] > 3:
        ax2.set_xlim(t[-1]-3, t[-1])
    ax2.set_ylim(min(omega2), max(omega2))

line_ani = animation.FuncAnimation(fig, update, frames=1000, interval=100)

plt.show()
