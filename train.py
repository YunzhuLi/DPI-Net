import os
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import DPINet
from data import PhysicsFleXDataset, collate_fn

from utils import count_parameters


parser = argparse.ArgumentParser()
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--time_step_clip', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./60.)
parser.add_argument('--nf_relation', type=int, default=300)
parser.add_argument('--nf_particle', type=int, default=200)
parser.add_argument('--nf_effect', type=int, default=200)
parser.add_argument('--env', default='')
parser.add_argument('--train_valid_ratio', type=float, default=0.9)
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--gen_data', type=int, default=0)
parser.add_argument('--gen_stat', type=int, default=0)
parser.add_argument('--log_per_iter', type=int, default=1000)
parser.add_argument('--ckp_per_iter', type=int, default=10000)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)

parser.add_argument('--n_instance', type=int, default=0)
parser.add_argument('--n_stages', type=int, default=0)
parser.add_argument('--n_his', type=int, default=0)

parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--forward_times', type=int, default=2)

parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--resume_iter', type=int, default=0)

# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)
parser.add_argument('--position_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

args = parser.parse_args()

phases_dict = dict()


if args.env == 'FluidFall':
    args.n_rollout = 3000

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [fluid]
    args.attr_dim = 1

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.time_step = 121
    args.time_step_clip = 5
    args.n_instance = 1
    args.n_stages = 1

    args.neighbor_radius = 0.08

    phases_dict["instance_idx"] = [0, 189]
    phases_dict["root_num"] = [[]]
    phases_dict["instance"] = ['fluid']
    phases_dict["material"] = ['fluid']

    args.outf = 'dump_FluidFall/' + args.outf

elif args.env == 'BoxBath':
    args.n_rollout = 3000

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [rigid, fluid, root_0]
    args.attr_dim = 3

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.time_step = 151
    args.time_step_clip = 0
    args.n_instance = 2
    args.n_stages = 4

    args.neighbor_radius = 0.08

    # ball, fluid
    phases_dict["instance_idx"] = [0, 64, 1024]
    phases_dict["root_num"] = [[8], []]
    phases_dict["root_sib_radius"] = [[0.4], []]
    phases_dict["root_des_radius"] = [[0.08], []]
    phases_dict["root_pstep"] = [[args.pstep], []]
    phases_dict["instance"] = ['cube', 'fluid']
    phases_dict["material"] = ['rigid', 'fluid']

    args.outf = 'dump_BoxBath/' + args.outf

elif args.env == 'FluidShake':
    args.n_rollout = 2000

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [fluid, wall_0, wall_1, wall_2, wall_3, wall_4]
    # wall_0: floor
    # wall_1: left
    # wall_2: right
    # wall_3: back
    # wall_4: front
    args.attr_dim = 6

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = 1
    args.time_step = 301
    args.time_step_clip = 0
    args.n_stages = 2

    args.neighbor_radius = 0.08

    phases_dict["root_num"] = [[]]
    phases_dict["instance"] = ["fluid"]
    phases_dict["material"] = ["fluid"]

    args.outf = 'dump_FluidShake/' + args.outf

elif args.env == 'RiceGrip':
    args.n_rollout = 5000
    args.n_his = 3

    # object state:
    # [rest_x, rest_y, rest_z, rest_xdot, rest_ydot, rest_zdot,
    #  x, y, z, xdot, ydot, zdot, quat.x, quat.y, quat.z, quat.w]
    args.state_dim = 16 + 6 * args.n_his
    args.position_dim = 6

    # object attr:
    # [fluid, root, gripper_0, gripper_1,
    #  clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep]
    args.attr_dim = 7

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = 1
    args.time_step = 41
    args.time_step_clip = 0
    args.n_stages = 4
    args.n_roots = 30

    args.neighbor_radius = 0.08

    phases_dict["root_num"] = [[args.n_roots]]
    phases_dict["root_sib_radius"] = [[5.0]]
    phases_dict["root_des_radius"] = [[0.2]]
    phases_dict["root_pstep"] = [[args.pstep]]
    phases_dict["instance"] = ["rice"]
    phases_dict["material"] = ["fluid"]

    args.outf = 'dump_RiceGrip/' + args.outf

else:
    raise AssertionError("Unsupported env")


args.outf = args.outf + '_' + args.env
args.dataf = 'data/' + args.dataf + '_' + args.env

os.system('mkdir -p ' + args.outf)
os.system('mkdir -p ' + args.dataf)

# generate data
datasets = {phase: PhysicsFleXDataset(
    args, phase, phases_dict, args.verbose_data) for phase in ['train', 'valid']}

for phase in ['train', 'valid']:
    if args.gen_data:
        datasets[phase].gen_data(args.env)
    else:
        datasets[phase].load_data(args.env)

use_gpu = torch.cuda.is_available()


dataloaders = {x: torch.utils.data.DataLoader(
    datasets[x], batch_size=args.batch_size,
    shuffle=True if x == 'train' else False,
    num_workers=args.num_workers,
    collate_fn=collate_fn)
    for x in ['train', 'valid']}


# define propagation network
model = DPINet(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)

print("Number of parameters: %d" % count_parameters(model))

if args.resume_epoch > 0 or args.resume_iter > 0:
    model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
    print("Loading saved ckp from %s" % model_path)
    model.load_state_dict(torch.load(model_path))

# criterion
criterionMSE = nn.MSELoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

if use_gpu:
    model = model.cuda()
    criterionMSE = criterionMSE.cuda()

st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
best_valid_loss = np.inf
for epoch in range(st_epoch, args.n_epoch):

    phases = ['train', 'valid'] if args.eval == 0 else ['valid']
    for phase in phases:

        model.train(phase=='train')

        losses = 0.
        for i, data in enumerate(dataloaders[phase]):

            attr, state, rels, n_particles, n_shapes, instance_idx, label = data
            Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

            Rr, Rs = [], []
            for j in range(len(rels[0])):
                Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]
                Rr.append(torch.sparse.FloatTensor(
                    Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
                Rs.append(torch.sparse.FloatTensor(
                    Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

            data = [attr, state, Rr, Rs, Ra, label]

            with torch.set_grad_enabled(phase=='train'):
                if use_gpu:
                    for d in range(len(data)):
                        if type(data[d]) == list:
                            for t in range(len(data[d])):
                                data[d][t] = Variable(data[d][t].cuda())
                        else:
                            data[d] = Variable(data[d].cuda())
                else:
                    for d in range(len(data)):
                        if type(data[d]) == list:
                            for t in range(len(data[d])):
                                data[d][t] = Variable(data[d][t])
                        else:
                            data[d] = Variable(data[d])

                attr, state, Rr, Rs, Ra, label = data

                # st_time = time.time()
                predicted = model(
                    attr, state, Rr, Rs, Ra, n_particles,
                    node_r_idx, node_s_idx, pstep,
                    instance_idx, phases_dict, args.verbose_model)
                # print('Time forward', time.time() - st_time)

                # print(predicted)
                # print(label)

            loss = criterionMSE(predicted, label)
            losses += np.sqrt(loss.item())

            if phase == 'train':
                if i % args.forward_times == 0:
                    # update parameters every args.forward_times
                    if i != 0:
                        loss_acc /= args.forward_times
                        optimizer.zero_grad()
                        loss_acc.backward()
                        optimizer.step()
                    loss_acc = loss
                else:
                    loss_acc += loss

            if i % args.log_per_iter == 0:
                n_relations = 0
                for j in range(len(Ra)):
                    n_relations += Ra[j].size(0)
                print('%s [%d/%d][%d/%d] n_relations: %d, Loss: %.6f, Agg: %.6f' %
                      (phase, epoch, args.n_epoch, i, len(dataloaders[phase]),
                       n_relations, np.sqrt(loss.item()), losses / (i + 1)))

            if phase == 'train' and i > 0 and i % args.ckp_per_iter == 0:
                torch.save(model.state_dict(), '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i))

        losses /= len(dataloaders[phase])
        print('%s [%d/%d] Loss: %.4f, Best valid: %.4f' %
              (phase, epoch, args.n_epoch, losses, best_valid_loss))

        if phase == 'valid':
            scheduler.step(losses)
            if(losses < best_valid_loss):
                best_valid_loss = losses
                torch.save(model.state_dict(), '%s/net_best.pth' % (args.outf))

