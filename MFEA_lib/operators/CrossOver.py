from re import U
from typing import Deque, Tuple
import numpy as np
import random

class AbstractCrossOver():
    def __init__(self, *args, **kwargs) -> None:
        pass
    def __call__(self, pa, pb, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

class SBX_CrossOver(AbstractCrossOver):
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nc = 15, *args, **kwargs):
        self.nc = nc
    def __call__(self, pa, pb, type = None, p_swap_inter = 0.1, d_swap = 0.1, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        '''
        type = 'inter' / 'intra' / ('inter1skf', p_swap_inter)
        '''
        assert type == 'inter' or type == 'intra' or type == 'inter1skf'

        u = np.random.rand(len(pa))

        # ~1
        beta = np.where(u < 0.5, (2*u)**(1/(self.nc +1)), (2 * (1 - u))**(-1 / (1 + self.nc)))
        
        #like pa
        c1 = 0.5*((1 + beta) * pa + (1 - beta) * pb)
        #like pb
        c2 = 0.5*((1 - beta) * pa + (1 + beta) * pb)

        c1, c2 = np.clip(c1, 0, 1), np.clip(c2, 0, 1)

        if type == 'intra':
            idx = np.where(np.random.rand(len(pa)) < 0.5)[0]
            c1[idx], c2[idx] = c2[idx], c1[idx]
            
        elif type == 'inter1skf':
            idx = np.where(np.random.rand(len(pa)) < p_swap_inter)[0]
            # idx = np.where((np.random.rand(len(pa)) < p_swap_inter) * (np.abs(c1 - c2) < d_swap))[0]
            # idx = np.where(np.abs(pa - pb) > d_swap)[0]
            c2[idx] = c1[idx]

        return c1, c2

class newSBX(AbstractCrossOver):
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nb_tasks: int, nc = 15, gamma = .9, num_save_mem = 1, *args, **kwargs):
        self.nc = nc
        self.nb_tasks = nb_tasks
        self.gamma = gamma
        self.nb_save_mem = num_save_mem
        self.nb_update = 0
    def get_dim_uss(self, dim_uss):
        self.dim_uss = dim_uss
        self.prob_crossover_dim = np.ones((self.nb_tasks, self.nb_tasks, dim_uss))
        
        #nb all offspring bored by crossover at dimensions d by task x task
        self.sum_crossover_each_dimensions = np.zeros((self.nb_tasks, self.nb_tasks, dim_uss))
        #index off offspring
        self.epoch_idx_crossover = []

        #nb inds alive after epoch
        self.success_crossover_each_dimension = np.zeros((self.nb_tasks, self.nb_tasks, dim_uss))
      
        self.skf_parent = np.empty((0, 2), dtype= int)

        #dis memory
        self.M_dis: list = []
        self.M_success_dis = [Deque(maxlen= self.nb_save_mem ) for i in range(self.nb_tasks)]
        
        # type_crossover: 'from_loc' / 'from_exp'
        self.type_crossover = []
        self.exp_rate = np.zeros((self.nb_tasks, self.nb_tasks)) + 0.5

    def update(self, idx_success):
        self.nb_update += 1
        count_type = np.zeros((self.nb_tasks, self.nb_tasks, 2))
        for idx in idx_success:
            if self.skf_parent[idx][0] != self.skf_parent[idx][1]:
                if self.type_crossover[idx] == 'from_loc':
                    # sum success crossover
                    self.success_crossover_each_dimension[self.skf_parent[idx][0], self.skf_parent[idx][1]] += self.epoch_idx_crossover[idx]
                    
                    count_type[self.skf_parent[0], self.skf_parent[1], 0] += 1
                else:
                    count_type[self.skf_parent[0], self.skf_parent[1], 1] += 1

        # update type_rate
        g = 0.7
        upd_rate = np.zeros_like(self.exp_rate)
        for i in range(self.nb_tasks):
            for j in range(self.nb_tasks):
                if count_type[i, j, 0] == 0 and count_type[i, j, 1] == 0:
                    upd_rate[i, j] = self.exp_rate[i, j]
                else:
                    upd_rate[i, j] = count_type[i, j, 1]/(count_type[i, j, 0] + count_type[i, j, 1] + 1e-2)

        self.exp_rate = g * self.exp_rate + (1 - g) * upd_rate

        # percent success:
        per_success = np.copy(self.prob_crossover_dim)
        per_success = np.where(
            self.sum_crossover_each_dimensions != 0, 
            self.success_crossover_each_dimension / (self.success_crossover_each_dimension + 1e-10),
            self.prob_crossover_dim
        )

        # update prob_crossover_dim 
        self.prob_crossover_dim = self.prob_crossover_dim * self.gamma + (1 - self.gamma) * per_success
        self.prob_crossover_dim = np.clip(self.prob_crossover_dim, 1/self.dim_uss, 1)

        # reset
        self.sum_crossover_each_dimensions = np.zeros((self.nb_tasks, self.nb_tasks, self.dim_uss))
        self.success_crossover_each_dimension = np.zeros((self.nb_tasks, self.nb_tasks, self.dim_uss))
        self.epoch_idx_crossover = []
        self.skf_parent = np.empty((0, 2), dtype= int)

        self.type_crossover = []

    def __call__(self, pa, pb, skf: tuple[int, int], *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        '''
        skf = (skf_pa, skf_pb)
        '''
        idx_crossover = (np.random.rand(self.dim_uss) < self.prob_crossover_dim[skf[0], skf[1]])
        self.sum_crossover_each_dimensions[skf[0], skf[1]] += 2 * idx_crossover
        self.epoch_idx_crossover.append(idx_crossover)
        self.epoch_idx_crossover.append(idx_crossover)
        self.skf_parent = np.append(self.skf_parent, [[skf[0], skf[1]]], axis = 0)
        self.skf_parent = np.append(self.skf_parent, [[skf[0], skf[1]]], axis = 0)
        # save dis of parents
        self.M_dis.append(pb - pa)
        self.M_dis.append(pb - pa)

        u = np.random.rand(self.dim_uss)    
        # ~1
        beta = np.where(u < 0.5, (2*u)**(1/(self.nc +1)), (2 * (1 - u))**(-1 / (1 + self.nc)))

        # intra
        if skf[0] == skf[1]:
            #like pa
            c1 = 0.5*((1 + beta) * pa + (1 - beta) * pb)
            #like pb
            c2 = 0.5*((1 - beta) * pa + (1 + beta) * pb)

            #swap
            idx = np.where(np.random.rand(len(pa)) < 0.5)[0]
            c1[idx], c2[idx] = c2[idx], c1[idx]
            
            self.type_crossover.append('from_loc')
            self.type_crossover.append('from_loc')
        else:

            # a learn from experience of b
            if np.random.rand() < self.exp_rate[skf[0], skf[1]] and len(self.M_success_dis[skf[1]]) > 0:
                pb_exp = pa - random.choice(self.M_success_dis[skf[1]])

                #like pa
                c1 = 0.5*((1 + beta) * pa + (1 - beta) * pb_exp)
                #like pb_exp
                c2 = 0.5*((1 - beta) * pa + (1 + beta) * pb_exp)

                self.type_crossover.append('from_exp')
                self.type_crossover.append('from_exp')

            # a learn from location of b
            else:
                #like pa
                c1 = np.where(idx_crossover, 0.5*((1 + beta) * pa + (1 - beta) * pb), pa)
                #like pb
                c2 = np.where(idx_crossover, 0.5*((1 - beta) * pa + (1 + beta) * pb), pa)

                self.type_crossover.append('from_loc')
                self.type_crossover.append('from_loc')

        c1, c2 = np.clip(c1, 0, 1), np.clip(c2, 0, 1)
        return c1, c2

    def get_eval(self, eval = (False, False)):
        if eval[0] == True:
            self.M_success_dis[self.skf_parent[-2][0]].append(self.M_dis[0])
        if eval[1] == True:
            self.M_success_dis[self.skf_parent[-1][0]].append(self.M_dis[1])
        self.M_dis = []