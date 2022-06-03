# Copyright 2020-present, Y. Gao
# All rights reserved.
#

import torch
import torch.nn as nn
import math
import torch.optim as optim
from .common import MLP, ResNet18

#auxiliary functions
def get_grad_vector(pp, grad_dims):
    """
     gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims))
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


def add_memory_grad(pp, mem_grads, grad_dims):
    """
        This stores the gradient of a new memory and compute the dot product with the previously stored memories.
        pp: parameters

        mem_grads: gradients of previous memories
        grad_dims: list with number of parameters per layers

    """

    # gather the gradient of the new memory
    grads = get_grad_vector(pp, grad_dims)

    if mem_grads is None:

        mem_grads = grads.unsqueeze(dim=0)


    else:

        grads = grads.unsqueeze(dim=0)

        mem_grads = torch.cat((mem_grads, grads), dim=0)

    return mem_grads

##################################################

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.rn = args.memory_strength# n the number of gradient vectors to estimate new samples similarity, line 5 in alg.2
        self.robust = args.robust
        self.nac_order = args.nac_order
        self.reg_factor = args.reg_factor
        self.is_cifar = ('cifar10' in args.data_file)
        if self.is_cifar:
            self.net = ResNet18(n_outputs, bias=args.bias)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.ce = nn.CrossEntropyLoss()
        self.ce_r = nn.CrossEntropyLoss(reduce=False)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.opt = torch.optim.Adam(self.parameters(), lr=args.lr)

        self.n_memories = args.n_memories# auxiliary storage before deciding samples to the buffer,
        # if this is equal to the batch size, then every batch the method decides which samples to add to the buffer.
        self.n_sampled_memories = args.n_sampled_memories #buffer size, M
        self.n_constraints = args.n_constraints #n_samples to be replayed from the buffer at each time a new batch is recieved, default equal to batch size
        self.gpu = args.cuda

        self.batch_size=args.batch_size
        self.n_iter = args.n_iter #number of iteraions (update steps) for each recieved batch
        self.sim_th = args.change_th  # cosine similarity threshold for being a candidate for buffer entrance
        # allocate ring buffer (default new batch size)
        self.memory_data = torch.FloatTensor(self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_memories)

        self.added_index = self.n_sampled_memories
        # allocate  buffer
        self.sampled_memory_data = None
        self.sampled_memory_labs = None
        self.sampled_effective_dim = None

        self.sampled_memory_cos = None# buffer cosine similarity score
        self.subselect=args.subselect# for estimating new samples score, draw samples in batch of size subselect
        # allocate selected constraints

        self.memory_size = (self.n_sampled_memories*self.n_inputs)
        self.memory_used = 0.0

        self.params = {n: p for n, p in self.named_parameters() if p.requires_grad}  # For convenience
        self.regularization_terms = {}

        # old grads to measure changes

        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())


        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0



    def forward(self, x, t=0):
        output = self.net(x)

        return output

    def cosine_similarity(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2

        cos = nn.CosineSimilarity(dim=1, eps=eps)

        sim = cos(x1, x2)

        return sim

    def calculate_importance(self, order=1):
        # Update the neuronal correlation for neuron importance
        print('Computing NAC')

        # Initialize the importance matrix
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized

        neuron_importance = {}
        norm_factor=0
        for n, p in importance.items():
            # weight in [N_l+1, N_l]
            weight = self.params[n].clone().detach()

            # we skip bias for now
            if len(weight.size())>1:
                w_hat = torch.abs(torch.tanh(weight.t()))
                # corr in [N_l, N_l]
                if order == 1:
                    # First order
                    w_corr = w_hat @ w_hat.t() / weight.size(0)
                elif order == 2:
                    # second order
                    w_corr = (w_hat @ w_hat.t()) * (w_hat @ w_hat.t()) / weight.size(0) ** 2

                # val, _ = torch.max(w_corr, 1)
                val = torch.mean(w_corr, 1)

                # find with normalize factor
                if torch.mean(val)>norm_factor:
                    norm_factor = torch.mean(val)
                neuron_importance[n] = val

        # normalize
        for n, p in importance.items():
            if len(self.params[n].size()) > 1:
                neuron_importance[n] = neuron_importance[n] / norm_factor

        layer_idx=0
        for n, p in importance.items():
            n_importance_list = list(neuron_importance.values())

            # middle layers
            if layer_idx<2:
                if len(p.size())>1:
                    # update on the weights
                    p += torch.ger(n_importance_list[layer_idx+1], n_importance_list[layer_idx])
                    # p += (n_importance_list[layer_idx + 1] + n_importance_list[layer_idx])/2

                else:
                    # update on bias
                    p += n_importance_list[layer_idx+1]
                    layer_idx += 1
            else:
                # last layer
                if len(p.size())>1:
                    # update on the weights
                    for row in p:
                        row += n_importance_list[layer_idx]
                    # use mean weight importance for the final layer bias
                    temp, _ = torch.max(p, 1)
                else:
                    # update on bias
                    p += temp

        return importance

    #print tasks and labels statistics of the selected buffer samples
    def print_taskids_stats(self):

        tasks=torch.unique(self.sampled_memory_taskids)
        for t in range(tasks.size(0)):
            print('task number ',tasks[t],'samples in buffer',torch.eq(self.sampled_memory_taskids,tasks[t]).nonzero().size(0))


    # MAIN TRAINING FUNCTION
    def observe(self, x, t, y):
        # update memory

        self.memory_data = torch.FloatTensor(self.n_memories, self.n_inputs).cuda()
        self.memory_labs = torch.LongTensor(self.n_memories).cuda()

        # Update ring buffer storing examples from current task, equals to batch size
        bsz = y.data.size(0)

        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz


        if self.sampled_memory_data is not None:
            #shuffle buffer, determine batch size of buffer sampled memories
            shuffeled_inds=torch.randperm(self.sampled_memory_labs.size(0))
            effective_batch_size = min(self.n_constraints, self.sampled_memory_labs.size(0))
            b_index=0
        #gradients of used buffer samples
        self.mem_grads = None

        this_sim=0

        for iter_i in range(self.n_iter):# number of iterations over a given batch of samples, i.e. number of update steps
            ###################################################################################################
            # now compute the grad on the current minibatch and perform update step on the newly recieved batch
            ###################################################################################################
            self.zero_grad()

            # efficiency regularization
            reg_loss = 0
            for param in self.parameters():
                if len(param.size()) > 1:
                    # l2,1 for first layer only
                    reg_loss = reg_loss + self.reg_factor * torch.sum(torch.norm(param, dim=0))
                    break

            out = self.forward(x)
            ce_loss = self.ce(out, y)
            # print(reg_loss.item())

            # robust training with NAC
            if self.robust > 0:
                for i, reg_term in self.regularization_terms.items():
                    task_reg_loss = 0
                    importance = reg_term['importance']
                    task_param = reg_term['task_param']
                    for n, p in self.params.items():
                        task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()

                    reg_loss += self.robust * task_reg_loss

            # print(reg_loss.item())

            loss = ce_loss + reg_loss
            loss.backward()
            # this_grad = get_grad_vector(self.parameters, self.grad_dims).unsqueeze(0)
            self.opt.step()

            # update memory
            self.memory_data = torch.FloatTensor(self.batch_size, self.n_inputs).cuda()
            self.memory_labs = torch.LongTensor(self.batch_size).cuda()

            self.memory_data.copy_(x.data)
            self.memory_labs.copy_(y.data)

            ###################################################################################################
            # update steps on the replayed samples from buffer, we only draw once
            ###################################################################################################
            if self.sampled_memory_data is not None:

                random_batch_inds = shuffeled_inds[b_index * effective_batch_size:b_index * effective_batch_size + effective_batch_size]
                batch_x = self.sampled_memory_data[random_batch_inds]
                batch_y = self.sampled_memory_labs[random_batch_inds]
                self.zero_grad()

                reg_loss = 0
                for param in self.parameters():
                    if len(param.size()) > 1:
                        # l2,1 for first layer only
                        reg_loss = reg_loss + self.reg_factor * torch.sum(torch.norm(param, dim=0))
                        break

                out = self.forward(batch_x)
                ce_loss = self.ce(out, batch_y)

                # robust training with NAC
                if self.robust > 0:
                    for i, reg_term in self.regularization_terms.items():
                        task_reg_loss = 0
                        importance = reg_term['importance']
                        task_param = reg_term['task_param']
                        for n, p in self.params.items():
                            task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()

                        reg_loss += self.robust * task_reg_loss

                loss = ce_loss + reg_loss
                loss.backward()

                self.opt.step()
                b_index += 1
                if b_index * effective_batch_size >= self.sampled_memory_labs.size(0):
                    b_index = 0

        ## compute r (also act as mask when storing the data)
        for param in self.parameters():
            r = torch.gt(torch.sum(torch.gt(torch.abs(param), 1e-2), dim=0), 0)
            break

        effective_dim = torch.sum(r)

        # print(effective_dim.item())
        # print(self.memory_used)
        # print(self.memory_size)

        ##HERE MEMORY IS EQUAL TO THE BATCH SIZE, this procedure is performed for every recieved batch
        if self.mem_cnt == self.n_memories :
            self.eval()

            if self.sampled_memory_data is not None and self.memory_size<self.memory_used + self.memory_data.size(0) * effective_dim:#buffer is full

                batch_sim=self.get_batch_sim(effective_batch_size)#estimate similarity score for the recieved samples to randomly drawn samples from buffer
                # for effecency we estimate the similarity for the whole batch

                if (batch_sim)<self.sim_th:

                    mem_data = self.memory_data.clone()
                    mem_lab = self.memory_labs.clone()

                    buffer_sim = (self.sampled_memory_cos - torch.min(self.sampled_memory_cos)) / ((torch.max(self.sampled_memory_cos) - torch.min(self.sampled_memory_cos)) + 0.01)

                    index=torch.multinomial(buffer_sim, mem_data.size(0), replacement=False)#draw candidates for replacement from the buffer

                    batch_item_sim=self.get_each_batch_sample_sim()# estimate the similarity of each sample in the recieved batch to the randomly drawn samples from the buffer.
                    scaled_batch_item_sim=((batch_item_sim+1)/2).unsqueeze(1).clone()
                    buffer_repl_batch_sim=((self.sampled_memory_cos[index]+1)/2).unsqueeze(1).clone()
                    #draw an event to decide on replacement decision
                    outcome=torch.multinomial(torch.cat((scaled_batch_item_sim,buffer_repl_batch_sim),dim=1), 1, replacement=False)#
                    #replace samples with outcome =1
                    added_indx = torch.arange(end=batch_item_sim.size(0))
                    sub_index=outcome.squeeze(1).byte()
                    self.sampled_memory_data[index[sub_index]] = mem_data[added_indx[sub_index]].clone()
                    self.sampled_memory_labs[index[sub_index]] = mem_lab[added_indx[sub_index]].clone()

                    # update memory_sued here, sub_index are those that been replaced
                    # release memory
                    self.memory_used = self.memory_used - torch.sum(self.sampled_effective_dim[index[sub_index]])

                    # update the new effective dim
                    self.sampled_effective_dim[index[sub_index]] = effective_dim.float()

                    # update the memory usage
                    self.memory_used = self.memory_used + torch.sum(self.sampled_effective_dim[index[sub_index]])

                    self.sampled_memory_cos[index[sub_index]] = batch_item_sim[added_indx[sub_index]].clone()
                    self.sampled_memory_taskids[index[sub_index]] = t

            else:
                #add new samples to the buffer
                added_inds = torch.arange(0, self.memory_data.size(0))

                new_task_ids = torch.zeros(added_inds.size(0)) + t

                new_effective_dims = torch.zeros(added_inds.size(0)) + effective_dim

                #first buffer insertion
                if self.sampled_memory_data is None:

                    self.sampled_memory_data = torch.mul(self.memory_data[added_inds].clone(), r.float())
                    self.sampled_memory_labs = self.memory_labs[added_inds].clone()

                    # update the new effective dim
                    self.sampled_effective_dim = new_effective_dims.clone()
                    # update the memory usage
                    self.memory_used = self.memory_used + torch.sum(self.sampled_effective_dim)

                    self.sampled_memory_taskids=new_task_ids.clone()

                    self.sampled_memory_cos=torch.zeros(added_inds.size(0)) + 0.1
                else:
                    self.get_batch_sim(effective_batch_size)#draw random samples from buffer
                    this_sampled_memory_cos = self.get_each_batch_sample_sim().clone()#estimate a score for each added sample
                    self.sampled_memory_cos = torch.cat((self.sampled_memory_cos, this_sampled_memory_cos.clone()),
                                                        dim=0)
                    self.sampled_memory_data = torch.cat((self.sampled_memory_data ,self.memory_data[added_inds].clone()),dim=0)
                    self.sampled_memory_labs = torch.cat(( self.sampled_memory_labs,self.memory_labs[added_inds].clone()),dim=0)

                    self.sampled_memory_taskids = torch.cat(( self.sampled_memory_taskids,new_task_ids),
                                                            dim=0).clone()

                    # update the new effective dim
                    self.sampled_effective_dim = torch.cat(( self.sampled_effective_dim, new_effective_dims),
                                                           dim=0).clone()
                    # update the memory usage
                    self.memory_used = self.memory_used + torch.sum(new_effective_dims)

            # self.print_taskids_stats()
            self.mem_cnt = 0
            self.train()

            if self.robust > 0:
                # Backup the weight of current task
                task_param = {}
                for n, p in self.params.items():
                    task_param[n] = p.clone().detach()

                # Calculate the importance of weights for current task
                importance = self.calculate_importance(order=self.nac_order)

                # Save the weight and importance of weights of current task
                # Always use only one slot in self.regularization_terms
                self.regularization_terms[1] = {'importance': importance, 'task_param': task_param}

    def get_batch_sim(self,effective_batch_size):

        b_index = 0
        self.mem_grads = None
        shuffeled_inds = torch.randperm(self.sampled_memory_labs.size(0))

        for iter_i in range(int(self.rn)):

            random_batch_inds = shuffeled_inds[
                                b_index * effective_batch_size:b_index * effective_batch_size + effective_batch_size]
            batch_x = self.sampled_memory_data[random_batch_inds]
            batch_y = self.sampled_memory_labs[random_batch_inds]
            b_index += 1
            self.zero_grad()
            out = self.forward(batch_x)
            loss = self.ce(out, batch_y)
            loss.backward()
            self.mem_grads = add_memory_grad(self.parameters, self.mem_grads, self.grad_dims)
            if b_index * effective_batch_size >= self.sampled_memory_labs.size(0):

                break

        self.zero_grad()
        out = self.forward(self.memory_data)
        loss = self.ce(out, self.memory_labs)
        loss.backward()
        this_grad = get_grad_vector(self.parameters, self.grad_dims).unsqueeze(0)
        batch_sim = max((self.cosine_similarity(self.mem_grads, this_grad)))

        return batch_sim

    def get_each_batch_sample_sim(self):
        cosine_sim = torch.zeros(self.memory_labs.size(0))
        item_index=0

        for x, y in zip(self.memory_data, self.memory_labs):
            self.zero_grad()
            out = self.forward(x.unsqueeze(0))
            ptloss = self.ce(out, y.unsqueeze(0))
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            this_grad = get_grad_vector(self.parameters, self.grad_dims).unsqueeze(0)

            cosine_sim[item_index]=max(self.cosine_similarity(self.mem_grads, this_grad))
            item_index+=1

        return cosine_sim
