from re import U
from typing import Tuple
import numpy as np


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
    def __init__(self, nb_tasks: int, nc = 15, gamma = .9, *args, **kwargs):
        self.nc = nc
        self.nb_tasks = nb_tasks
        self.gamma = gamma

    def get_dim_uss(self, dim_uss):
        self.dim_uss = dim_uss
        self.prob = np.ones((self.nb_tasks, self.nb_tasks, dim_uss))
        
        #nb all offspring bored by crossover at dimensions d by task x task
        self.sum_crossover_each_dimensions = np.zeros((self.nb_tasks, self.nb_tasks, dim_uss))
        #index off offspring
        self.epoch_idx_crossover = []

        #nb inds alive after epoch
        self.success_crossover_each_dimension = np.zeros((self.nb_tasks, self.nb_tasks, dim_uss))
      
        self.skf_parent = np.empty((0, 2), dtype= int)

    def update(self, idx_success):
        # sum success crossover
        for idx in idx_success:
            self.success_crossover_each_dimension[self.skf_parent[idx][0], self.skf_parent[idx][1]] += self.epoch_idx_crossover[idx]

        # percent success:
        per_success = np.copy(self.prob)
        per_success = np.where(
            self.sum_crossover_each_dimensions != 0, 
            self.success_crossover_each_dimension / (self.success_crossover_each_dimension + 1e-10),
            self.prob
        )

        # update prob 
        self.prob = self.prob * self.gamma + (1 - self.gamma) * per_success

        # reset
        self.sum_crossover_each_dimensions = np.zeros((self.nb_tasks, self.nb_tasks, self.dim_uss))
        self.success_crossover_each_dimension = np.zeros((self.nb_tasks, self.nb_tasks, self.dim_uss))
        self.epoch_idx_crossover = []
        self.skf_parent = np.empty((0, 2), dtype= int)

        self.prob = np.clip(self.prob, 1/self.dim_uss, 1)
        
    def __call__(self, pa, pb, skf: tuple[int, int], *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        '''
        skf = (skf_pa, skf_pb)
        '''
        idx_crossover = (np.random.rand(self.dim_uss) < self.prob[skf[0], skf[1]]) 
        self.sum_crossover_each_dimensions[skf[0], skf[1]] += 2 * idx_crossover
        self.epoch_idx_crossover.append(idx_crossover)
        self.epoch_idx_crossover.append(idx_crossover)
        self.skf_parent = np.append(self.skf_parent, [[skf[0], skf[1]]], axis = 0)
        self.skf_parent = np.append(self.skf_parent, [[skf[0], skf[1]]], axis = 0)

        u = np.random.rand(self.dim_uss)    
        # ~1
        beta = np.where(u < 0.5, (2*u)**(1/(self.nc +1)), (2 * (1 - u))**(-1 / (1 + self.nc)))

        if skf[0] == skf[1]:
            #like pa
            c1 = 0.5*((1 + beta) * pa + (1 - beta) * pb)
            #like pb
            c2 = 0.5*((1 - beta) * pa + (1 + beta) * pb)

            c1, c2 = np.clip(c1, 0, 1), np.clip(c2, 0, 1)

            #swap
            idx = np.where(np.random.rand(len(pa)) < 0.5)[0]
            c1[idx], c2[idx] = c2[idx], c1[idx]
        
        else:
            #like pa
            c1 = np.where(idx_crossover, 0.5*((1 + beta) * pa + (1 - beta) * pb), pa)
            #like pb
            c2 = np.where(idx_crossover, 0.5*((1 - beta) * pa + (1 + beta) * pb), pa)

            c1, c2 = np.clip(c1, 0, 1), np.clip(c2, 0, 1)

            # NOTE swap
        return c1, c2