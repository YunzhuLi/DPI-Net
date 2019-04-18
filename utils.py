from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import h5py
import os

import torch
from torch.autograd import Variable


def rand_int(lo, hi):
    return np.random.randint(lo, hi)

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_variable(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(),
                        requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor),
                        requires_grad=requires_grad)


def visualize_point_clouds(point_clouds, c=['b', 'r'], view=None, store=False, store_path=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    frame.axes.zaxis.set_ticklabels([])

    for i in range(len(point_clouds)):
        points = point_clouds[i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c[i], s=10, alpha=0.3)

    X, Y, Z = point_clouds[0][:, 0], point_clouds[0][:, 1], point_clouds[0][:, 2]

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.grid(False)

    if view is None:
        view = 0, 0
    ax.view_init(view[0], view[1])
    plt.show()

    # plt.pause(5)

    if store:
        fig.savefig(store_path, bbox_inches='tight')


def sample_control_RiceGrip():
    dis = np.random.rand() * 0.5
    angle = np.random.rand() * np.pi * 2.
    x = np.cos(angle) * dis
    z = np.sin(angle) * dis
    d = np.random.rand() * 0.3 + 0.7    # (0.6, 0.9)
    return x, z, d


def sample_control_FluidShake(x_box, time_step, dt):
    control = np.zeros(time_step)
    v_box = 0.
    for step in range(time_step):
        control[step] = v_box
        x_box += v_box * dt
        v_box += rand_float(-0.15, 0.15) - x_box * 0.1
    return control


def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat


def quatFromAxisAngle_var(axis, angle):
    axis /= torch.norm(axis)

    half = angle * 0.5
    w = torch.cos(half)

    sin_theta_over_two = torch.sin(half)
    axis *= sin_theta_over_two

    quat = torch.cat([axis, w])
    # print("quat size", quat.size())

    return quat


def calc_shape_states_RiceGrip(t, dt, shape_state_dim, gripper_config):
    rest_gripper_dis = 1.8
    x, z, d = gripper_config
    s = (rest_gripper_dis - d) / 2.
    half_rest_gripper_dis = rest_gripper_dis / 2.

    time = max(0., t) * 5
    lastTime = max(0., t - dt) * 5

    states = np.zeros((2, shape_state_dim))

    dis = np.sqrt(x**2 + z**2)
    angle = np.array([-z / dis, x / dis])
    quat = quatFromAxisAngle(np.array([0., 1., 0.]), np.arctan(x / z))

    e_0 = np.array([x + z * half_rest_gripper_dis / dis, z - x * half_rest_gripper_dis / dis])
    e_1 = np.array([x - z * half_rest_gripper_dis / dis, z + x * half_rest_gripper_dis / dis])

    e_0_curr = e_0 + angle * np.sin(time) * s
    e_1_curr = e_1 - angle * np.sin(time) * s
    e_0_last = e_0 + angle * np.sin(lastTime) * s
    e_1_last = e_1 - angle * np.sin(lastTime) * s

    states[0, :3] = np.array([e_0_curr[0], 0.6, e_0_curr[1]])
    states[0, 3:6] = np.array([e_0_last[0], 0.6, e_0_last[1]])
    states[0, 6:10] = quat
    states[0, 10:14] = quat

    states[1, :3] = np.array([e_1_curr[0], 0.6, e_1_curr[1]])
    states[1, 3:6] = np.array([e_1_last[0], 0.6, e_1_last[1]])
    states[1, 6:10] = quat
    states[1, 10:14] = quat

    return states


def calc_shape_states_RiceGrip_var(t, dt, gripper_config):
    rest_gripper_dis = Variable(torch.FloatTensor([1.8]).cuda())
    x, z, d = gripper_config[0:1], gripper_config[1:2], gripper_config[2:3]

    s = (rest_gripper_dis - d) / 2.
    half_rest_gripper_dis = rest_gripper_dis / 2.

    time = max(0., t) * 5
    lastTime = max(0., t - dt) * 5

    dis = torch.sqrt(x**2 + z**2)
    angle = torch.cat([-z / dis, x / dis])
    quat = quatFromAxisAngle_var(Variable(torch.FloatTensor([0., 1., 0.]).cuda()), torch.atan(x / z))

    e_0 = torch.cat([x + z * half_rest_gripper_dis / dis, z - x * half_rest_gripper_dis / dis])
    e_1 = torch.cat([x - z * half_rest_gripper_dis / dis, z + x * half_rest_gripper_dis / dis])

    e_0_curr = e_0 + angle * np.sin(time) * s
    e_1_curr = e_1 - angle * np.sin(time) * s
    e_0_last = e_0 + angle * np.sin(lastTime) * s
    e_1_last = e_1 - angle * np.sin(lastTime) * s

    y = Variable(torch.FloatTensor([0.6]).cuda())
    states_0 = torch.cat([e_0_curr[0:1], y, e_0_curr[1:2], e_0_last[0:1], y, e_0_last[1:2], quat, quat])
    states_1 = torch.cat([e_1_curr[0:1], y, e_1_curr[1:2], e_1_last[0:1], y, e_1_last[1:2], quat, quat])

    # print(states_0.requires_grad, states_1.requires_grad)
    # print("gripper #0:", states_0.size())
    # print("gripper #1:", states_1.size())

    return torch.cat([states_0.view(1, -1), states_1.view(1, -1)], 0)


def calc_box_init_FluidShake(dis_x, dis_z, height, border):
    center = np.array([0., 0., 0.])
    quat = np.array([1., 0., 0., 0.])
    boxes = []

    # floor
    halfEdge = np.array([dis_x/2., border/2., dis_z/2.])
    boxes.append([halfEdge, center, quat])

    # left wall
    halfEdge = np.array([border/2., (height+border)/2., dis_z/2.])
    boxes.append([halfEdge, center, quat])

    # right wall
    boxes.append([halfEdge, center, quat])

    # back wall
    halfEdge = np.array([(dis_x+border*2)/2., (height+border)/2., border/2.])
    boxes.append([halfEdge, center, quat])

    # front wall
    boxes.append([halfEdge, center, quat])

    return boxes


def calc_shape_states_FluidShake(x_curr, x_last, box_dis, height, border):
    dis_x, dis_z = box_dis
    quat = np.array([1., 0., 0., 0.])

    states = np.zeros((5, 14))

    states[0, :3] = np.array([x_curr, border/2., 0.])
    states[0, 3:6] = np.array([x_last, border/2., 0.])

    states[1, :3] = np.array([x_curr-(dis_x+border)/2., (height+border)/2., 0.])
    states[1, 3:6] = np.array([x_last-(dis_x+border)/2., (height+border)/2., 0.])

    states[2, :3] = np.array([x_curr+(dis_x+border)/2., (height+border)/2., 0.])
    states[2, 3:6] = np.array([x_last+(dis_x+border)/2., (height+border)/2., 0.])

    states[3, :3] = np.array([x_curr, (height+border)/2., -(dis_z+border)/2.])
    states[3, 3:6] = np.array([x_last, (height+border)/2., -(dis_z+border)/2.])

    states[4, :3] = np.array([x_curr, (height+border)/2., (dis_z+border)/2.])
    states[4, 3:6] = np.array([x_last, (height+border)/2., (dis_z+border)/2.])

    states[:, 6:10] = quat
    states[:, 10:] = quat

    return states


def calc_shape_states_FluidShake_var(x_curr, x_last, box_dis, height, border):
    dis_x, dis_z = box_dis

    dis_x = Variable(torch.FloatTensor([dis_x]).cuda())
    dis_z = Variable(torch.FloatTensor([dis_z]).cuda())
    height = Variable(torch.FloatTensor([height]).cuda())
    border = Variable(torch.FloatTensor([border]).cuda())
    zero = Variable(torch.FloatTensor([0.]).cuda())
    quat = Variable(torch.FloatTensor([1., 0., 0., 0.]).cuda())

    state_0 = torch.cat([
        x_curr, border/2., zero, x_last, border/2., zero, quat, quat]).view(1, -1)

    state_1 = torch.cat([
        x_curr-(dis_x+border)/2., (height+border)/2., zero,
        x_last-(dis_x+border)/2., (height+border)/2., zero,
        quat, quat]).view(1, -1)

    state_2 = torch.cat([
        x_curr+(dis_x+border)/2., (height+border)/2., zero,
        x_last+(dis_x+border)/2., (height+border)/2., zero,
        quat, quat]).view(1, -1)

    state_3 = torch.cat([
        x_curr, (height+border)/2., -(dis_z+border)/2.,
        x_last, (height+border)/2., -(dis_z+border)/2.,
        quat, quat]).view(1, -1)

    state_4 = torch.cat([
        x_curr, (height+border)/2., (dis_z+border)/2.,
        x_last, (height+border)/2., (dis_z+border)/2.,
        quat, quat]).view(1, -1)

    states = torch.cat([state_0, state_1, state_2, state_3, state_4], 0)
    # print("states size", states.size())

    return states


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [N, D]
        # y: [M, D]
        x = x.repeat(y.size(0), 1, 1)   # x: [M, N, D]
        x = x.transpose(0, 1)           # x: [N, M, D]
        y = y.repeat(x.size(0), 1, 1)   # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)    # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])   # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        return self.chamfer_distance(pred, label)
