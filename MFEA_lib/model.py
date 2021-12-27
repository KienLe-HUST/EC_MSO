from typing import List
import numpy as np

from .func import AbstractFunc
from .GA import population_init, factorial_cost, factorial_rank, skill_factor_best_task, polynomial_mutation, sbx_crossover

class AbstractModel():
    def __init__(self) -> None:
        pass
    def compile(self, cross_over, mutation, selection):
        self.cross_over = cross_over
        self.mutation = mutation
        self.selection = selection
        pass
    def fit(self, tasks: List[AbstractFunc], num_generations, num_inds_each_task = 100, range_init_pop = [0, 1], evaluate_initial_skillFactor = True,one_line = False, num_epochs_printed = 20):
        # initial history of factorial cost -> for render
        history_cost = np.empty((0, len(tasks)), np.float) 

        # dim of Unified search space
        dim_uss = max([t.dim for t in tasks])

        # initial population
        inf, sup = range_init_pop
        population, skill_factor_arr = population_init(
            N = num_inds_each_task, 
            num_tasks = len(tasks), 
            d = dim_uss,
            min_val = inf,
            max_val = sup,
        )
        if evaluate_initial_skillFactor:
            skill_factor_arr = skill_factor_best_task(population, tasks)
        
        pop_fcost = factorial_cost(population, skill_factor_arr, tasks)
        
        history_cost = np.append(history_cost, 
            [[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))]], 
            axis = 0
        )

        for epoch in range(num_generations):
            
            # initial offspring of generation
            offspring = np.empty((0, dim_uss))
            offspring_skill_factor = np.empty((0, ), np.int)
            offspring_fcost = np.empty((0, ))

            while len(offspring) < len(population):
                [idx_a, idx_b] = np.random.choice(len(population), size= 2, replace= False)
                [pa, pb], [skf_a, skf_b] = population[[idx_a, idx_b]], skill_factor_arr[[idx_a, idx_b]]

                if skf_a == skf_b:
                    # intra - crossover
                    ca, cb = self.cross_over(pa, pb)