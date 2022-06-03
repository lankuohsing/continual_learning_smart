
import argparse
import os.path
import torch
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--i', default='raw/', help='input directory')
parser.add_argument('--o', default='mnist_split', help='output file')
parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--ratio', default=0, type=float, help='corruption ratio')
args = parser.parse_args()

torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr = torch.load(os.path.join(args.i, 'mnist_train.pt'))
x_te, y_te = torch.load(os.path.join(args.i, 'mnist_test.pt'))
x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
x_te = x_te.float().view(x_te.size(0), -1) / 255.0
y_tr = y_tr.view(-1).long()
y_te = y_te.view(-1).long()

# prepare label corruption
if args.ratio > 0:
    corrupt_idx = torch.FloatTensor(y_tr.size()).uniform_() < args.ratio
    csize = torch.sum(corrupt_idx)
    label_idxs = torch.arange(10).long()
    for idx in range(len(corrupt_idx)):
        if corrupt_idx[idx]:
            orig_label_idx = y_tr[idx]
            temp = label_idxs[label_idxs != orig_label_idx]
            corrupted_label = torch.multinomial(temp.float(), 1)
            y_tr[idx] = corrupted_label

cpt = int(10 / args.n_tasks)

for t in range(args.n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt
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

    tasks_tr.append([(c1, c2), x_tr[i_tr[:1000]].clone(), y_tr[i_tr[:1000]].clone()])
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])

if args.ratio > 0:
    out_dir = args.o + '_corrupt_' + str(args.ratio) + '.pt'
else:
    out_dir = args.o + '.pt'

torch.save([tasks_tr, tasks_te], out_dir)
