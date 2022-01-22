import numpy as np
from ..GA import factorial_rank

class AbstractSelection():
    def __init__(self):
        pass
    def __call__(self, skill_factor_arr, pop_fitness, nb_inds_tasks: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

class ElitismSelection(AbstractSelection):
    '''
    `skill_factor_arr`: current skill factor of population
    `pop_fitness`: population's scalar_fitness
    `nb_inds_tasks`: num individuals of tasks

    return: index of selected individuals
    '''
    def __init__(self,):
        super().__init__()
    def __call__(self, skill_factor_arr, pop_fitness, nb_inds_tasks: np.ndarray, shuffle = True, *args, **kwargs) -> np.ndarray:
        idx_selected_inds = np.empty((0,), dtype= int)
        
        for i in range (len(nb_inds_tasks)):
            idx_inds_i = np.where(skill_factor_arr == i)[0]

            N_i = min(np.int(nb_inds_tasks[i]), len(idx_inds_i))
            
            sorted_idx = idx_inds_i[np.argsort(-pop_fitness[idx_inds_i])]
            idx_selected_inds = np.append(idx_selected_inds, sorted_idx[:N_i], axis = 0)
        if shuffle:
            np.random.shuffle(idx_selected_inds)
        return idx_selected_inds