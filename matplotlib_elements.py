import matplotlib.patches as patches
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class Spring(Line2D):
    def __init__(self, ax, free_length, coil_number=3, linewidth=None, color=None):
        super().__init__([], [], linewidth=linewidth, color=color) 
        self.end_length = free_length * 0.2
        self.coil_width = free_length / 30
        self.coil_number = coil_number
        ax.add_line(self)

    def set_ends(self, point1, point2):
        assert len(point1) == 2 and len(point2) == 2
        point1, point2 = np.array(point1), np.array(point2)
        u = point2 - point1 
        length = np.linalg.norm(u)
        u = u / length
        v = np.array([-u[1], u[0]])

        xs, ys = [], []
        xs.append(point1[0])
        ys.append(point1[1])
        p = point1 + u * (self.end_length/2)
        xs.append(p[0])
        ys.append(p[1])

        pitch = (length - self.end_length) / self.coil_number
        p -= u * (pitch/4)
        for k in range(self.coil_number * 2):
            direc = 1 if k%2==0 else -1
            p += u * (pitch/2)
            p += v * self.coil_width * direc
            xs.append(p[0])
            ys.append(p[1])
            p -= v * self.coil_width * direc

        p += u * (pitch/4)
        xs.append(p[0])
        ys.append(p[1])
        xs.append(point2[0])
        ys.append(point2[1])
        super().set_data(xs, ys)


class Rectangle(patches.Rectangle):
    def __init__(self, ax, width, height, color):
        super().__init__((0.0,0.0), width, height, facecolor=color)
        self.width = width 
        self.height = height 
        ax.add_patch(self)

    def set_center(self, x, y):
        super().set_xy(( x-self.width/2, y-self.height/2 ))


class BallStick(Line2D):
    def __init__(self, ax, ball_size=None, ball_color=None, stick_width=None, stick_color=None):
        super().__init__([], [], linewidth=stick_width, color=stick_color)
        self.ball = Line2D([], [], color=ball_color, markersize=ball_size, marker='o')
        ax.add_line(self)
        ax.add_line(self.ball)

    def set_ends(self, p1, p2):
        assert len(p1) == 2 and len(p1) == 2
        super().set_data(*zip(p1, p2))
        self.ball.set_data([p2[0]], [p2[1]])

    def draw(self, renderer):
        super().draw(renderer)
        self.ball.draw(renderer)


class Stick(Line2D):
    def __init__(self, ax, stick_width=None, stick_color=None):
        super().__init__([], [], linewidth=stick_width, color=stick_color)
        ax.add_line(self)

    def set_ends(self, p1, p2):
        assert len(p1) == 2 and len(p1) == 2
        super().set_data(*zip(p1, p2))


class MovableLine3DCollection(Line3DCollection):
    def __init__(self, ax3d, verts, vert_connections, linewidths=None, colors=None):
        self.verts = np.array(verts)
        self.vert_connections = vert_connections
        
        segments = []
        for vert_connection in self.vert_connections:
            segments.append(self.verts[vert_connection, :])
        super().__init__(segments, linewidths=linewidths, colors=colors)
        ax3d.add_collection3d(self)

    def rotate(self, rotation_center, rotation_axis, rotation_angle):
        rotation_center = np.array([rotation_center])
        e_z = np.array(rotation_axis) / np.linalg.norm(rotation_axis)
        e_x = np.array([-e_z[1], e_z[0], 0])
        if np.linalg.norm(e_x) == 0:
            e_x = np.array([-e_z[2], 0, e_z[0]])
        e_x = e_x / np.linalg.norm(e_x)
        e_y = np.cross(e_z, e_x)
        Q = np.stack([e_x, e_y, e_z], axis=-1)
        R = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
             [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
             [                      0,                      0, 1]])
        T = Q @ R @ Q.transpose()
        self.verts = (self.verts-rotation_center) @ T.transpose() + rotation_center

        segments = []
        for vert_connection in self.vert_connections:
            segments.append(self.verts[vert_connection, :])
        super().set_segments(segments)
        return self.verts

    def translate(self, translation):
        translation = np.array([translation])
        self.verts = self.verts + translation

        segments = []
        for vert_connection in self.vert_connections:
            segments.append(self.verts[vert_connection, :])
        super().set_segments(segments)
        return self.verts

