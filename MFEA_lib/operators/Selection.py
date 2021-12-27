import numpy as np
from ..GA import factorial_rank

class AbstractSelection():
    def __init__(self):
        pass
    def __call__(self) -> np.ndarray:
        pass

class ElitismSelection(AbstractSelection):
    '''
    return index of selected individuals
    '''
    def __call__(self, nb_each_task, skill_factor_arr, pop_fcost, nb_tasks) -> np.ndarray:
        pop_finess = 1/factorial_rank(pop_fcost, skill_factor_arr, nb_tasks)
        num_inds_pop = np.int(nb_each_task) * nb_tasks
        idx_selected_inds = np.argsort(-pop_finess)[:num_inds_pop]
        return idx_selected_inds