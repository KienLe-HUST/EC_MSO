import numpy as np
from .GA import population_init, factorial_cost, factorial_rank, skill_factor_best_task, polynomial_mutation, sbx_crossover

class AbstractModel():
    def __init__(self, num_epochs, num_inds_each_task, range_init_pop = [0, 1],  
                one_line = False, num_epochs_printed = 20, evaluate_initial_skillFactor = True) -> None:
        self.num_epochs = num_epochs
        self.num_inds_each_task = num_inds_each_task
        self.range_init_pop = range_init_pop
        self.one_line = one_line
        self.num_epochs_printed = num_epochs_printed
        self.evaluate_initial_skillFactor = evaluate_initial_skillFactor
    def compile(self):
        pass
    def fit(self, tasks = []):
        # initial history of factorial cost -> for render
        history_cost = np.empty((0, len(tasks)), np.float) 

        max_d = max([t.dim for t in tasks])
        
        