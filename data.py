import os
import torch
import time
import random
import numpy as np
import gzip
import pickle
import h5py

import multiprocessing as mp
import scipy.spatial as spatial
from sklearn.cluster import MiniBatchKMeans

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from utils import rand_float, rand_int
from utils import sample_control_RiceGrip, calc_shape_states_RiceGrip
from utils import calc_box_init_FluidShake, calc_shape_states_FluidShake


def collate_fn(data):
    return data[0]


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0**2 * n_0 + std_1**2 * n_1 + \
                   (mean_0 - mean)**2 * n_0 + (mean_1 - mean)**2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def normalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.
            s = Variable(torch.FloatTensor(stat[i]).cuda())

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].size(1) / stat_dim)
            data[i] = data[i].view(-1, n_rep, stat_dim)

            data[i] = (data[i] - s[:, 0]) / s[:, 1]

            data[i] = data[i].view(-1, n_rep * stat_dim)

    else:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].shape[1] / stat_dim)
            data[i] = data[i].reshape((-1, n_rep, stat_dim))

            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]

            data[i] = data[i].reshape((-1, n_rep * stat_dim))

    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]

    return data


def rotateByQuat(p, quat):
    R = np.zeros((3, 3))
    a, b, c, d = quat[3], quat[0], quat[1], quat[2]
    R[0, 0] = a**2 + b**2 - c**2 - d**2
    R[0, 1] = 2 * b * c - 2 * a * d
    R[0, 2] = 2 * b * d + 2 * a * c
    R[1, 0] = 2 * b * c + 2 * a * d
    R[1, 1] = a**2 - b**2 + c**2 - d**2
    R[1, 2] = 2 * c * d - 2 * a * b
    R[2, 0] = 2 * b * d - 2 * a * c
    R[2, 1] = 2 * c * d + 2 * a * b
    R[2, 2] = a**2 - b**2 - c**2 + d**2

    return np.dot(R, p)


def gen_PyFleX(info):

    env, root_num = info['env'], info['root_num']
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, n_instance = info['n_rollout'], info['n_instance']
    time_step, time_step_clip = info['time_step'], info['time_step_clip']
    shape_state_dim, dt = info['shape_state_dim'], info['dt']

    env_idx = info['env_idx']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2**32)

    # positions, velocities
    if env_idx == 5:    # RiceGrip
        stats = [init_stat(6), init_stat(6)]
    else:
        stats = [init_stat(3), init_stat(3)]

    import pyflex
    pyflex.init()

    for i in range(n_rollout):

        if i % 10 == 0:
            print("%d / %d" % (i, n_rollout))

        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)

        if env == 'FluidFall':
            scene_params = np.zeros(1)
            pyflex.set_scene(env_idx, scene_params, thread_idx)
            n_particles = pyflex.get_n_particles()
            positions = np.zeros((time_step, n_particles, 3), dtype=np.float32)
            velocities = np.zeros((time_step, n_particles, 3), dtype=np.float32)

            for j in range(time_step_clip):
                p_clip = pyflex.get_positions().reshape(-1, 4)[:, :3]
                pyflex.step()

            for j in range(time_step):
                positions[j] = pyflex.get_positions().reshape(-1, 4)[:, :3]

                if j == 0:
                    velocities[j] = (positions[j] - p_clip) / dt
                else:
                    velocities[j] = (positions[j] - positions[j - 1]) / dt

                pyflex.step()

                data = [positions[j], velocities[j]]
                store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

        elif env == 'BoxBath':
            # BoxBath

            scene_params = np.zeros(1)
            pyflex.set_scene(env_idx, scene_params, thread_idx)
            n_particles = pyflex.get_n_particles()
            positions = np.zeros((time_step, n_particles, 3), dtype=np.float32)
            velocities = np.zeros((time_step, n_particles, 3), dtype=np.float32)

            for j in range(time_step_clip):
                pyflex.step()

            p = pyflex.get_positions().reshape(-1, 4)[:64, :3]
            clusters = []
            st_time = time.time()
            kmeans = MiniBatchKMeans(n_clusters=root_num[0][0], random_state=0).fit(p)
            # print('Time on kmeans', time.time() - st_time)
            clusters.append([[kmeans.labels_]])
            # centers = kmeans.cluster_centers_

            ref_rigid = p

            for j in range(time_step):
                positions[j] = pyflex.get_positions().reshape(-1, 4)[:, :3]

                # apply rigid projection to ground truth
                XX = ref_rigid
                YY = positions[j, :64]
                # print("MSE init", np.mean(np.square(XX - YY)))

                X = XX.copy().T
                Y = YY.copy().T
                mean_X = np.mean(X, 1, keepdims=True)
                mean_Y = np.mean(Y, 1, keepdims=True)
                X = X - mean_X
                Y = Y - mean_Y
                C = np.dot(X, Y.T)
                U, S, Vt = np.linalg.svd(C)
                D = np.eye(3)
                D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
                R = np.dot(Vt.T, np.dot(D, U.T))
                t = mean_Y - np.dot(R, mean_X)

                YY_fitted = (np.dot(R, XX.T) + t).T
                # print("MSE fit", np.mean(np.square(YY_fitted - YY)))

                positions[j, :64] = YY_fitted

                if j > 0:
                    velocities[j] = (positions[j] - positions[j - 1]) / dt

                pyflex.step()

                data = [positions[j], velocities[j], clusters]
                store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

        elif env == 'FluidShake':
            # if env is FluidShake
            height = 1.0
            border = 0.025
            dim_x = rand_int(10, 12)
            dim_y = rand_int(15, 20)
            dim_z = 3
            x_center = rand_float(-0.2, 0.2)
            x = x_center - (dim_x-1)/2.*0.055
            y = 0.055/2. + border + 0.01
            z = 0. - (dim_z-1)/2.*0.055
            box_dis_x = dim_x * 0.055 + rand_float(0., 0.3)
            box_dis_z = 0.2

            scene_params = np.array([x, y, z, dim_x, dim_y, dim_z, box_dis_x, box_dis_z])
            pyflex.set_scene(env_idx, scene_params, 0)

            boxes = calc_box_init_FluidShake(box_dis_x, box_dis_z, height, border)

            for i in range(len(boxes)):
                halfEdge = boxes[i][0]
                center = boxes[i][1]
                quat = boxes[i][2]
                pyflex.add_box(halfEdge, center, quat)

            n_particles = pyflex.get_n_particles()
            n_shapes = pyflex.get_n_shapes()

            # print("n_particles", n_particles)
            # print("n_shapes", n_shapes)

            positions = np.zeros((time_step, n_particles + n_shapes, 3), dtype=np.float32)
            velocities = np.zeros((time_step, n_particles + n_shapes, 3), dtype=np.float32)
            shape_quats = np.zeros((time_step, n_shapes, 4), dtype=np.float32)

            x_box = x_center
            v_box = 0.
            for j in range(time_step_clip):
                x_box_last = x_box
                x_box += v_box * dt
                shape_states_ = calc_shape_states_FluidShake(
                    x_box, x_box_last, scene_params[-2:], height, border)
                pyflex.set_shape_states(shape_states_)
                pyflex.step()

            for j in range(time_step):
                x_box_last = x_box
                x_box += v_box * dt
                v_box += rand_float(-0.15, 0.15) - x_box * 0.1
                shape_states_ = calc_shape_states_FluidShake(
                    x_box, x_box_last, scene_params[-2:], height, border)
                pyflex.set_shape_states(shape_states_)

                positions[j, :n_particles] = pyflex.get_positions().reshape(-1, 4)[:, :3]
                shape_states = pyflex.get_shape_states().reshape(-1, shape_state_dim)

                for k in range(n_shapes):
                    positions[j, n_particles + k] = shape_states[k, :3]
                    shape_quats[j, k] = shape_states[k, 6:10]

                if j > 0:
                    velocities[j] = (positions[j] - positions[j - 1]) / dt

                pyflex.step()

                data = [positions[j], velocities[j], shape_quats[j], scene_params]
                store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

        elif env == 'RiceGrip':
            # if env is RiceGrip
            # repeat the grip for R times
            R = 3
            gripper_config = sample_control_RiceGrip()

            if i % R == 0:
                ### set scene
                # x, y, z: [8.0, 10.0]
                # clusterStiffness: [0.3, 0.7]
                # clusterPlasticThreshold: [0.00001, 0.0005]
                # clusterPlasticCreep: [0.1, 0.3]
                x = rand_float(8.0, 10.0)
                y = rand_float(8.0, 10.0)
                z = rand_float(8.0, 10.0)

                clusterStiffness = rand_float(0.3, 0.7)
                clusterPlasticThreshold = rand_float(0.00001, 0.0005)
                clusterPlasticCreep = rand_float(0.1, 0.3)

                scene_params = np.array([x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])
                pyflex.set_scene(env_idx, scene_params, thread_idx)
                scene_params[4] *= 1000.

                halfEdge = np.array([0.15, 0.8, 0.15])
                center = np.array([0., 0., 0.])
                quat = np.array([1., 0., 0., 0.])
                pyflex.add_box(halfEdge, center, quat)
                pyflex.add_box(halfEdge, center, quat)

                n_particles = pyflex.get_n_particles()
                n_shapes = pyflex.get_n_shapes()

                positions = np.zeros((time_step, n_particles + n_shapes, 6), dtype=np.float32)
                velocities = np.zeros((time_step, n_particles + n_shapes, 6), dtype=np.float32)
                shape_quats = np.zeros((time_step, n_shapes, 4), dtype=np.float32)

                for j in range(time_step_clip):
                    shape_states = calc_shape_states_RiceGrip(0, dt, shape_state_dim, gripper_config)
                    pyflex.set_shape_states(shape_states)
                    pyflex.step()

                p = pyflex.get_positions().reshape(-1, 4)[:, :3]

                clusters = []
                st_time = time.time()
                kmeans = MiniBatchKMeans(n_clusters=root_num[0][0], random_state=0).fit(p)
                # print('Time on kmeans', time.time() - st_time)
                clusters.append([[kmeans.labels_]])
                # centers = kmeans.cluster_centers_

            for j in range(time_step):
                shape_states = calc_shape_states_RiceGrip(j * dt, dt, shape_state_dim, gripper_config)
                pyflex.set_shape_states(shape_states)

                positions[j, :n_particles, :3] = pyflex.get_rigidGlobalPositions().reshape(-1, 3)
                positions[j, :n_particles, 3:] = pyflex.get_positions().reshape(-1, 4)[:, :3]
                shape_states = pyflex.get_shape_states().reshape(-1, shape_state_dim)

                for k in range(n_shapes):
                    positions[j, n_particles + k, :3] = shape_states[k, :3]
                    positions[j, n_particles + k, 3:] = shape_states[k, :3]
                    shape_quats[j, k] = shape_states[k, 6:10]

                if j > 0:
                    velocities[j] = (positions[j] - positions[j - 1]) / dt

                pyflex.step()

                data = [positions[j], velocities[j], shape_quats[j], clusters, scene_params]
                store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

        else:
            raise AssertionError("Unsupported env")

        # change dtype for more accurate stat calculation
        # only normalize positions and velocities
        datas = [positions.astype(np.float64), velocities.astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0] * datas[j].shape[1]
            stats[j] = combine_stat(stats[j], stat)

    pyflex.clean()

    return stats


def visualize_neighbors(anchors, queries, idx, neighbors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(queries[idx, 0], queries[idx, 1], queries[idx, 2], c='g', s=80)
    ax.scatter(anchors[neighbors, 0], anchors[neighbors, 1], anchors[neighbors, 2], c='r', s=80)
    ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], alpha=0.2)
    ax.set_aspect('equal')

    plt.show()


def find_relations_neighbor(positions, query_idx, anchor_idx, radius, order, var=False):
    if np.sum(anchor_idx) == 0:
        return []

    pos = positions.data.cpu().numpy() if var else positions

    point_tree = spatial.cKDTree(pos[anchor_idx])
    neighbors = point_tree.query_ball_point(pos[query_idx], radius, p=order)

    '''
    for i in range(len(neighbors)):
        visualize_neighbors(pos[anchor_idx], pos[query_idx], i, neighbors[i])
    '''

    relations = []
    for i in range(len(neighbors)):
        count_neighbors = len(neighbors[i])
        if count_neighbors == 0:
            continue

        receiver = np.ones(count_neighbors, dtype=np.int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i]])

        # receiver, sender, relation_type
        relations.append(np.stack([receiver, sender, np.ones(count_neighbors)], axis=1))

    return relations


def make_hierarchy(env, attr, positions, velocities, idx, st, ed, phases_dict, count_nodes, clusters, verbose=0, var=False):
    order = 2
    n_root_level = len(phases_dict["root_num"][idx])
    attr, relations, node_r_idx, node_s_idx, pstep = [attr], [], [], [], []

    relations_rev, node_r_idx_rev, node_s_idx_rev, pstep_rev = [], [], [], []

    pos = positions.data.cpu().numpy() if var else positions
    vel = velocities.data.cpu().numpy() if var else velocities

    for i in range(n_root_level):
        root_num = phases_dict["root_num"][idx][i]
        root_sib_radius = phases_dict["root_sib_radius"][idx][i]
        root_des_radius = phases_dict["root_des_radius"][idx][i]
        root_pstep = phases_dict["root_pstep"][idx][i]

        if verbose:
            print('root info', root_num, root_sib_radius, root_des_radius, root_pstep)


        ### clustring the nodes
        # st_time = time.time()
        # kmeans = MiniBatchKMeans(n_clusters=root_num, random_state=0).fit(pos[st:ed, 3:6])
        # print('Time on kmeans', time.time() - st_time)
        # clusters = kmeans.labels_
        # centers = kmeans.cluster_centers_


        ### relations between roots and desendants
        rels, rels_rev = [], []
        node_r_idx.append(np.arange(count_nodes, count_nodes + root_num))
        node_s_idx.append(np.arange(st, ed))
        node_r_idx_rev.append(node_s_idx[-1])
        node_s_idx_rev.append(node_r_idx[-1])
        pstep.append(1); pstep_rev.append(1)

        if verbose:
            centers = np.zeros((root_num, 3))
            for j in range(root_num):
                des = np.nonzero(clusters[i][0]==j)[0]
                center = np.mean(pos[st:ed][des, -3:], 0, keepdims=True)
                centers[j] = center[0]
                visualize_neighbors(pos[st:ed], center, 0, des)

        for j in range(root_num):
            desendants = np.nonzero(clusters[i][0]==j)[0]
            roots = np.ones(desendants.shape[0]) * j
            if verbose:
                print(roots, desendants)
            rels += [np.stack([roots, desendants, np.zeros(desendants.shape[0])], axis=1)]
            rels_rev += [np.stack([desendants, roots, np.zeros(desendants.shape[0])], axis=1)]
            if verbose:
                print(np.max(np.sqrt(np.sum(np.square(pos[st + desendants, :3] - centers[j]), 1))))

        relations.append(np.concatenate(rels, 0))
        relations_rev.append(np.concatenate(rels_rev, 0))


        ### relations between roots and roots
        # point_tree = spatial.cKDTree(centers)
        # neighbors = point_tree.query_ball_point(centers, root_sib_radius, p=order)

        '''
        for j in range(len(neighbors)):
            visualize_neighbors(centers, centers, j, neighbors[j])
        '''

        rels = []
        node_r_idx.append(np.arange(count_nodes, count_nodes + root_num))
        node_s_idx.append(np.arange(count_nodes, count_nodes + root_num))
        pstep.append(root_pstep)

        roots = np.repeat(np.arange(root_num), root_num)
        siblings = np.tile(np.arange(root_num), root_num)
        if verbose:
            print(roots, siblings)
        rels += [np.stack([roots, siblings, np.zeros(root_num * root_num)], axis=1)]
        if verbose:
            print(np.max(np.sqrt(np.sum(np.square(centers[siblings, :3] - centers[j]), 1))))

        relations.append(np.concatenate(rels, 0))


        ### add to attributes/positions/velocities
        positions = [positions]
        velocities = [velocities]
        attributes = []
        for j in range(root_num):
            ids = np.nonzero(clusters[i][0]==j)[0]
            if var:
                positions += [torch.mean(positions[0][st:ed, :][ids], 0, keepdim=True)]
                velocities += [torch.mean(velocities[0][st:ed, :][ids], 0, keepdim=True)]
            else:
                positions += [np.mean(positions[0][st:ed, :][ids], 0, keepdims=True)]
                velocities += [np.mean(velocities[0][st:ed, :][ids], 0, keepdims=True)]

            attributes += [np.mean(attr[0][st:ed, :][ids], 0, keepdims=True)]

        attributes = np.concatenate(attributes, 0)

        if env == 'BoxBath':
            attributes[:, 2 + i] = 1
        elif env == 'RiceGrip':
            attributes[:, 1 + i] = 1

        if verbose:
            print('Attr sum', np.sum(attributes, 0))

        attr += [attributes]
        if var:
            positions = torch.cat(positions, 0)
            velocities = torch.cat(velocities, 0)
        else:
            positions = np.concatenate(positions, 0)
            velocities = np.concatenate(velocities, 0)

        st = count_nodes
        ed = count_nodes + root_num
        count_nodes += root_num

        if verbose:
            print(st, ed, count_nodes, positions.shape, velocities.shape)

    attr = np.concatenate(attr, 0)
    if verbose:
        print("attr", attr.shape)

    relations += relations_rev[::-1]
    node_r_idx += node_r_idx_rev[::-1]
    node_s_idx += node_s_idx_rev[::-1]
    pstep += pstep_rev[::-1]

    return attr, positions, velocities, count_nodes, relations, node_r_idx, node_s_idx, pstep


def prepare_input(data, stat, args, phases_dict, verbose=0, var=False):

    # Arrangement:
    # particles, shapes, roots

    if args.env == 'RiceGrip':
        positions, velocities, shape_quats, clusters, scene_params = data
        n_shapes = shape_quats.size(0) if var else shape_quats.shape[0]
    elif args.env == 'FluidShake':
        positions, velocities, shape_quats, scene_params = data
        n_shapes = shape_quats.size(0) if var else shape_quats.shape[0]
        clusters = None
    elif args.env == 'BoxBath':
        positions, velocities, clusters = data
        n_shapes = 0
    elif args.env == 'FluidFall':
        positions, velocities = data
        n_shapes = 0
        clusters = None

    count_nodes = positions.size(0) if var else positions.shape[0]
    n_particles = count_nodes - n_shapes

    if verbose:
        print("positions", positions.shape)
        print("velocities", velocities.shape)

        print("n_particles", n_particles)
        print("n_shapes", n_shapes)
        if args.env == 'RiceGrip' or args.env == 'FluidShake':
            print("shape_quats", shape_quats.shape)

    ### instance idx
    #   instance_idx (n_instance + 1): start idx of instance
    if args.env == 'RiceGrip' or args.env == 'FluidShake':
        instance_idx = [0, n_particles]
    else:
        instance_idx = phases_dict["instance_idx"]
    if verbose:
        print("Instance_idx:", instance_idx)


    ### object attributes
    #   dim 10: [rigid, fluid, root_0, root_1, gripper_0, gripper_1, mass_inv,
    #            clusterStiffness, clusterPlasticThreshold, cluasterPlasticCreep]
    attr = np.zeros((count_nodes, args.attr_dim))
    # no need to include mass for now
    # attr[:, 6] = positions[:, -1].data.cpu().numpy() if var else positions[:, -1] # mass_inv
    if args.env == 'RiceGrip':
        # clusterStiffness, clusterPlasticThreshold, cluasterPlasticCreep
        attr[:, -3:] = scene_params[-3:]


    ### construct relations
    Rr_idxs = []        # relation receiver idx list
    Rs_idxs = []        # relation sender idx list
    Ras = []            # relation attributes list
    values = []         # relation value list (should be 1)
    node_r_idxs = []    # list of corresponding receiver node idx
    node_s_idxs = []    # list of corresponding sender node idx
    psteps = []         # propagation steps

    ##### add env specific graph components
    rels = []
    if args.env == 'RiceGrip':
        # nodes = np.arange(n_particles)
        for i in range(n_shapes):
            attr[n_particles + i, 2 + i] = 1

            pos = positions.data.cpu().numpy() if var else positions
            dis = np.linalg.norm(
                pos[:n_particles, 3:6:2] - pos[n_particles + i, 3:6:2], axis=1)
            nodes = np.nonzero(dis < 0.3)[0]

            if verbose:
                visualize_neighbors(positions, positions, 0, nodes)
                print(np.sort(dis)[:10])

            gripper = np.ones(nodes.shape[0], dtype=np.int) * (n_particles + i)
            rels += [np.stack([nodes, gripper, np.ones(nodes.shape[0])], axis=1)]

    elif args.env == 'FluidShake':
        for i in range(n_shapes):
            attr[n_particles + i, 1 + i] = 1

            pos = positions.data.cpu().numpy() if var else positions
            if i == 0:
                # floor
                dis = pos[:n_particles, 1] - pos[n_particles + i, 1]
            elif i == 1:
                # left
                dis = pos[:n_particles, 0] - pos[n_particles + i, 0]
            elif i == 2:
                # right
                dis = pos[n_particles + i, 0] - pos[:n_particles, 0]
            elif i == 3:
                # back
                dis = pos[:n_particles, 2] - pos[n_particles + i, 2]
            elif i == 4:
                # front
                dis = pos[n_particles + i, 2] - pos[:n_particles, 2]
            else:
                raise AssertionError("more shapes than expected")
            nodes = np.nonzero(dis < 0.1)[0]

            if verbose:
                visualize_neighbors(positions, positions, 0, nodes)
                print(np.sort(dis)[:10])

            wall = np.ones(nodes.shape[0], dtype=np.int) * (n_particles + i)
            rels += [np.stack([nodes, wall, np.ones(nodes.shape[0])], axis=1)]

    if verbose and len(rels) > 0:
        print(np.concatenate(rels, 0).shape)

    ##### add relations between leaf particles
    for i in range(len(instance_idx) - 1):
        st, ed = instance_idx[i], instance_idx[i + 1]

        if verbose:
            print('instance #%d' % i, st, ed)

        if args.env == 'BoxBath':
            if phases_dict['material'][i] == 'rigid':
                attr[st:ed, 0] = 1
                queries = np.arange(st, ed)
                anchors = np.concatenate((np.arange(st), np.arange(ed, n_particles)))
            elif phases_dict['material'][i] == 'fluid':
                attr[st:ed, 1] = 1
                queries = np.arange(st, ed)
                anchors = np.arange(n_particles)
            else:
                raise AssertionError("Unsupported materials")

        elif args.env == 'FluidFall' or args.env == 'RiceGrip' or args.env == 'FluidShake':
            if phases_dict['material'][i] == 'fluid':
                attr[st:ed, 0] = 1
                queries = np.arange(st, ed)
                anchors = np.arange(n_particles)
            else:
                raise AssertionError("Unsupported materials")

        else:
            raise AssertionError("Unsupported materials")

        # st_time = time.time()
        pos = positions
        pos = pos[:, -3:]
        rels += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)
        # print("Time on neighbor search", time.time() - st_time)

    if verbose:
        print("Attr shape (after add env specific graph components):", attr.shape)
        print("Object attr:", np.sum(attr, axis=0))

    rels = np.concatenate(rels, 0)
    if rels.shape[0] > 0:
        if verbose:
            print("Relations neighbor", rels.shape)
        Rr_idxs.append(torch.LongTensor([rels[:, 0], np.arange(rels.shape[0])]))
        Rs_idxs.append(torch.LongTensor([rels[:, 1], np.arange(rels.shape[0])]))
        Ra = np.zeros((rels.shape[0], args.relation_dim))
        Ras.append(torch.FloatTensor(Ra))
        values.append(torch.FloatTensor([1] * rels.shape[0]))
        node_r_idxs.append(np.arange(n_particles))
        node_s_idxs.append(np.arange(n_particles + n_shapes))
        psteps.append(args.pstep)

    if verbose:
        print('clusters', clusters)

    # add heirarchical relations per instance
    cnt_clusters = 0
    for i in range(len(instance_idx) - 1):
        st, ed = instance_idx[i], instance_idx[i + 1]
        n_root_level = len(phases_dict["root_num"][i])

        if n_root_level > 0:
            attr, positions, velocities, count_nodes, \
            rels, node_r_idx, node_s_idx, pstep = \
                    make_hierarchy(args.env, attr, positions, velocities, i, st, ed,
                                   phases_dict, count_nodes, clusters[cnt_clusters], verbose, var)

            for j in range(len(rels)):
                if verbose:
                    print("Relation instance", j, rels[j].shape)
                Rr_idxs.append(torch.LongTensor([rels[j][:, 0], np.arange(rels[j].shape[0])]))
                Rs_idxs.append(torch.LongTensor([rels[j][:, 1], np.arange(rels[j].shape[0])]))
                Ra = np.zeros((rels[j].shape[0], args.relation_dim)); Ra[:, 0] = 1
                Ras.append(torch.FloatTensor(Ra))
                values.append(torch.FloatTensor([1] * rels[j].shape[0]))
                node_r_idxs.append(node_r_idx[j])
                node_s_idxs.append(node_s_idx[j])
                psteps.append(pstep[j])

            cnt_clusters += 1

    if verbose:
        if args.env == 'RiceGrip' or args.env == 'FluidShake':
            print("Scene_params:", scene_params)

        print("Attr shape (after hierarchy building):", attr.shape)
        print("Object attr:", np.sum(attr, axis=0))
        print("Particle attr:", np.sum(attr[:n_particles], axis=0))
        print("Shape attr:", np.sum(attr[n_particles:n_particles+n_shapes], axis=0))
        print("Roots attr:", np.sum(attr[n_particles+n_shapes:], axis=0))

    ### normalize data
    data = [positions, velocities]
    data = normalize(data, stat, var)
    positions, velocities = data[0], data[1]

    if verbose:
        print("Particle positions stats")
        print(positions.shape)
        print(np.min(positions[:n_particles], 0))
        print(np.max(positions[:n_particles], 0))
        print(np.mean(positions[:n_particles], 0))
        print(np.std(positions[:n_particles], 0))

        show_vel_dim = 6 if args.env == 'RiceGrip' else 3
        print("Velocities stats")
        print(velocities.shape)
        print(np.mean(velocities[:n_particles, :show_vel_dim], 0))
        print(np.std(velocities[:n_particles, :show_vel_dim], 0))

    if args.env == 'RiceGrip':
        if var:
            quats = torch.cat(
                [Variable(torch.zeros(n_particles, 4).cuda()), shape_quats,
                 Variable(torch.zeros(count_nodes - n_particles - n_shapes, 4).cuda())], 0)
            state = torch.cat([positions, velocities, quats], 1)
        else:
            quat_null = np.array([[0., 0., 0., 0.]])
            quats = np.repeat(quat_null, [count_nodes], axis=0)
            quats[n_particles:n_particles + n_shapes] = shape_quats
            # if args.eval == 0:
            # quats += np.random.randn(quats.shape[0], 4) * 0.05
            state = torch.FloatTensor(np.concatenate([positions, velocities, quats], axis=1))
    else:
        if var:
            state = torch.cat([positions, velocities], 1)
        else:
            state = torch.FloatTensor(np.concatenate([positions, velocities], axis=1))

    if verbose:
        for i in range(count_nodes - 1):
            if np.sum(np.abs(attr[i] - attr[i + 1])) > 1e-6:
                print(i, attr[i], attr[i + 1])

        for i in range(len(Ras)):
            print(i, np.min(node_r_idxs[i]), np.max(node_r_idxs[i]), np.min(node_s_idxs[i]), np.max(node_s_idxs[i]))

    attr = torch.FloatTensor(attr)
    relations = [Rr_idxs, Rs_idxs, values, Ras, node_r_idxs, node_s_idxs, psteps]

    return attr, state, relations, n_particles, n_shapes, instance_idx


class PhysicsFleXDataset(Dataset):

    def __init__(self, args, phase, phases_dict, verbose):
        self.args = args
        self.phase = phase
        self.phases_dict = phases_dict
        self.verbose = verbose
        self.data_dir = os.path.join(self.args.dataf, phase)
        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')

        os.system('mkdir -p ' + self.data_dir)

        if args.env == 'RiceGrip':
            self.data_names = ['positions', 'velocities', 'shape_quats', 'clusters', 'scene_params']
        elif args.env == 'FluidShake':
            self.data_names = ['positions', 'velocities', 'shape_quats', 'scene_params']
        elif args.env == 'BoxBath':
            self.data_names = ['positions', 'velocities', 'clusters']
        elif args.env == 'FluidFall':
            self.data_names = ['positions', 'velocities']

        ratio = self.args.train_valid_ratio
        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

    def __len__(self):
        return self.n_rollout * (self.args.time_step - 1)

    def load_data(self, name):
        self.stat = load_data(self.data_names[:2], self.stat_path)
        for i in range(len(self.stat)):
            self.stat[i] = self.stat[i][-self.args.position_dim:, :]
            # print(self.data_names[i], self.stat[i].shape)

    def gen_data(self, name):
        # if the data hasn't been generated, generate the data
        print("Generating data ... n_rollout=%d, time_step=%d" % (self.n_rollout, self.args.time_step))

        infos = []
        for i in range(self.args.num_workers):
            info = {
                'env': self.args.env,
                'root_num': self.phases_dict['root_num'],
                'thread_idx': i,
                'data_dir': self.data_dir,
                'data_names': self.data_names,
                'n_rollout': self.n_rollout // self.args.num_workers,
                'n_instance': self.args.n_instance,
                'time_step': self.args.time_step,
                'time_step_clip': self.args.time_step_clip,
                'dt': self.args.dt,
                'shape_state_dim': self.args.shape_state_dim}

            if self.args.env == 'BoxBath':
                info['env_idx'] = 1
            elif self.args.env == 'FluidFall':
                info['env_idx'] = 4
            elif self.args.env == 'RiceGrip':
                info['env_idx'] = 5
            elif self.args.env == 'FluidShake':
                info['env_idx'] = 6
            else:
                raise AssertionError("Unsupported env")

            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)
        data = pool.map(gen_PyFleX, infos)

        print("Training data generated, warpping up stats ...")

        if self.phase == 'train' and self.args.gen_stat:
            # positions [x, y, z], velocities[xdot, ydot, zdot]
            if self.args.env == 'RiceGrip':
                self.stat = [init_stat(6), init_stat(6)]
            else:
                self.stat = [init_stat(3), init_stat(3)]
            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])
            store_data(self.data_names[:2], self.stat, self.stat_path)
        else:
            print("Loading stat from %s ..." % self.stat_path)
            self.stat = load_data(self.data_names[:2], self.stat_path)

    def __getitem__(self, idx):
        idx_rollout = idx // (self.args.time_step - 1)
        idx_timestep = idx % (self.args.time_step - 1)

        # ignore the first frame for env RiceGrip
        if self.args.env == 'RiceGrip' and idx_timestep == 0:
            idx_timestep = np.random.randint(1, self.args.time_step - 1)

        data_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep) + '.h5')
        data_nxt_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep + 1) + '.h5')

        data = load_data(self.data_names, data_path)

        vel_his = []
        for i in range(self.args.n_his):
            path = os.path.join(self.data_dir, str(idx_rollout), str(max(1, idx_timestep - i - 1)) + '.h5')
            data_his = load_data(self.data_names, path)
            vel_his.append(data_his[1])

        data[1] = np.concatenate([data[1]] + vel_his, 1)

        attr, state, relations, n_particles, n_shapes, instance_idx = \
                prepare_input(data, self.stat, self.args, self.phases_dict, self.verbose)

        ### label
        data_nxt = normalize(load_data(self.data_names, data_nxt_path), self.stat)

        label = torch.FloatTensor(data_nxt[1][:n_particles])

        return attr, state, relations, n_particles, n_shapes, instance_idx, label

