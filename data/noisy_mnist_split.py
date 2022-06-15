# -*- coding: utf-8 -*-


import argparse
import os.path
import torch

# In[]
parser = argparse.ArgumentParser()

parser.add_argument('--i', default='raw/', help='input directory')
parser.add_argument('--o', default='noisy_mnist_split', help='output file')
parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--ratio', default=0, type=float, help='corruption ratio')
args = parser.parse_args()

torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr = torch.load(os.path.join(args.i, 'noisy_mnist_train.pt'))# (60000,28,28)
x_te, y_te = torch.load(os.path.join(args.i, 'noisy_mnist_test.pt'))
# In[]
x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0# (60000,784)
x_te = x_te.float().view(x_te.size(0), -1) / 255.0
y_tr = y_tr.view(-1).long()
y_te = y_te.view(-1).long()

# In[]
"""
[
    [top_row,left_col],
    [top_row,right_col],
    [bottom_row,left_col],
    [bottom_row,right_col]
]
"""
annot_coords_tr = torch.load(os.path.join(args.i, 'annot_coords_tr.pt'))
annot_coords_te = torch.load(os.path.join(args.i, 'annot_coords_te.pt'))

# In[]
class_num_per_task = int(10 / args.n_tasks)

for t in range(args.n_tasks):# default 5
    c1 = t * class_num_per_task
    c2 = (t + 1) * class_num_per_task
    i_tr_pos = (y_tr == c1).nonzero().view(-1)
    i_tr_neg = (y_tr == c1 + 1).nonzero().view(-1)
    i_te_pos = (y_te == c1).nonzero().view(-1)
    i_te_neg = (y_te == c1 + 1).nonzero().view(-1)

    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)

    y_tr[i_tr_pos] = 1
    y_tr[i_tr_neg] = 0
    y_te[i_te_pos] = 1
    y_te[i_te_neg] = 0

    tasks_tr.append([(c1, c2), x_tr[i_tr[:1000]].clone(), y_tr[i_tr[:1000]].clone(),  annot_coords_tr[i_tr[:1000]].clone()])
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone(), annot_coords_te[i_te[:1000]].clone()])
# In[]
out_dir = args.o + '.pt'
torch.save([tasks_tr, tasks_te], out_dir)