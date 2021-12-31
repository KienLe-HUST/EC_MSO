from typing import Tuple
import numpy as np
from ..operators import CrossOver, Mutation, Selection
from ..tasks.function import AbstractFunc
from ..GA import population_init, factorial_cost, factorial_rank, skill_factor_best_task
import sys
import matplotlib.pyplot as plt

class AbstractModel():

    def __init__(self) -> None:
        self.history_cost: np.ndarray
        self.solve: List[np.ndarray]
        pass
    def render(self, shape: Tuple[int, int], title = "", yscale = None, ylim: list[float, float] = None):
        fig = plt.figure(figsize= (shape[1]* 6, shape[0] * 5))
        fig.suptitle(title, size = 20)
        fig.set_facecolor("white")

        for i in range(self.history_cost.shape[1]):
            plt.subplot(shape[0], shape[1], i+1)

            plt.plot(np.arange(self.history_cost.shape[0]), self.history_cost[:, i])

            plt.title(self.tasks[i].name)
            plt.xlabel("Generations")
            plt.ylabel("Factorial Cost")
            
            if yscale is not None:
                plt.yscale(yscale)
            if ylim is not None:
                plt.ylim(bottom = ylim[0], top = ylim[1])
                
        plt.show()
        return fig
    def save(self, PATH):
        pass
    def compile(self, cross_over: CrossOver.AbstractCrossOver, mutation: Mutation.AbstractMutation, selection: Selection.AbstractSelection):
        self.cross_over = cross_over
        self.mutation = mutation
        self.selection = selection      
    def fit(self, tasks: list[AbstractFunc], num_generations, num_inds_each_task = 100,  evaluate_initial_skillFactor = True,
            range_init_pop = [0, 1], log_oneline = False, num_epochs_printed = 20) -> tuple[list[np.ndarray], np.ndarray]:
        assert num_generations > num_epochs_printed
        self.tasks = tasks

        # initial history of factorial cost -> for render
        self.history_cost = np.empty((0, len(tasks)), np.float) 

class MFEA_base(AbstractModel):

    def compile(self, cross_over = CrossOver.SBX_CrossOver(), mutation=  Mutation.Polynomial_Mutation(), selection= Selection.ElitismSelection()):
        super().compile(cross_over, mutation, selection)

    def fit(self, tasks: list[AbstractFunc], num_generations, num_inds_each_task = 100, rmp = 0.3, range_init_pop = [0, 1], evaluate_initial_skillFactor = True,
            log_oneline = False, num_epochs_printed = 20) -> tuple[list[np.ndarray], np.ndarray]:
        
        super().fit(tasks, num_generations, num_inds_each_task=num_inds_each_task, range_init_pop=range_init_pop, evaluate_initial_skillFactor=evaluate_initial_skillFactor, 
                    log_oneline= log_oneline, num_epochs_printed=num_epochs_printed)

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
        
        self.history_cost = np.append(self.history_cost, 
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
                [pa, pb], [skf_pa, skf_pb] = population[[idx_a, idx_b]], skill_factor_arr[[idx_a, idx_b]]

                if skf_pa == skf_pb or np.random.uniform() < rmp:
                    # intra, inter - crossover
                    oa, ob = self.cross_over(pa, pb)
                    skf_oa, skf_ob = np.random.choice([skf_pa, skf_pb], size= 2, replace= True)
                else:
                    # mutation
                    oa = self.mutation(pa)
                    ob = self.mutation(pb)
                    skf_oa, skf_ob = skf_pa, skf_pb

                offspring = np.append(offspring, [oa, ob], axis = 0)
                offspring_skill_factor = np.append(offspring_skill_factor, [skf_oa, skf_ob], axis = 0)
            
            offspring_fcost = factorial_cost(offspring, offspring_skill_factor, tasks)

            # merge
            population = np.append(population, offspring, axis = 0)
            skill_factor_arr = np.append(skill_factor_arr, offspring_skill_factor, axis = 0)
            pop_fcost = np.append(pop_fcost, offspring_fcost, axis = 0)

            # selection
            pop_fitness = 1/factorial_rank(pop_fcost, skill_factor_arr, len(tasks))
            idx = self.selection(skill_factor_arr, pop_fitness, [num_inds_each_task] * len(tasks))

            population = population[idx]
            skill_factor_arr = skill_factor_arr[idx]
            pop_fcost = pop_fcost[idx]

            #save history
            self.history_cost = np.append(self.history_cost, 
                [[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))]], 
                axis = 0
            )

            #print
            if (epoch + 1) % (num_generations // num_epochs_printed) == 0:
                if log_oneline == True:
                    sys.stdout.write('\r')
                sys.stdout.write('Epoch [{}/{}], [%-20s] %3d%% ,func_val: {}'
                    .format(epoch + 1, num_generations,[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))])
                    % ('=' * ((epoch + 1) // (num_generations // 20)) + '>' , (epoch + 1) * 100 // num_generations)
                    )
                if log_oneline == False:
                    print("\n")
                sys.stdout.flush()
        print('End')

        #solve
        sol_idx = [np.argmin(pop_fcost[np.where(skill_factor_arr == idx)]) for idx in range (len(tasks))]
        self.solve = [task.decode(population[np.where(skill_factor_arr == idx)][sol_idx[idx]]) for idx, task in enumerate(tasks)]

        return self.solve, self.history_cost

class MFEA1(AbstractModel):
    def compile(self, cross_over = CrossOver.SBX_CrossOver(), mutation=  Mutation.Polynomial_Mutation(), selection= Selection.ElitismSelection()):
        return super().compile(cross_over, mutation, selection)
    
    def fit(self, tasks: list[AbstractFunc], num_generations, num_inds_each_task=100, rmp = 0.3, range_init_pop=[0, 1], evaluate_initial_skillFactor=True, 
            log_oneline =False, num_epochs_printed=20) -> tuple[list[np.ndarray], np.ndarray]:   
                
        super().fit(tasks, num_generations, num_inds_each_task=num_inds_each_task, range_init_pop=range_init_pop, evaluate_initial_skillFactor=evaluate_initial_skillFactor, 
                    log_oneline= log_oneline, num_epochs_printed=num_epochs_printed)
        
        # dim of Unified search space
        dim_uss = max([t.dim for t in tasks]) 

        # initial population
        inf, sup = range_init_pop

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
        
        self.history_cost = np.append(self.history_cost, 
            [[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))]], 
            axis = 0
        )

        for epoch in range(num_generations):
            # initial offspring of generation
            offspring = np.empty((0, dim_uss))
            offspring_skill_factor = np.empty((0, ), np.int)
            offspring_fcost = np.empty((0, ))

            while len(offspring) < len(population):
                [idx_pa, idx_pb] = np.random.choice(len(population), size= 2, replace= False)
                [pa, pb], [skf_pa, skf_pb] = population[[idx_pa, idx_pb]], skill_factor_arr[[idx_pa, idx_pb]]

                if skf_pa == skf_pb:
                    # Intra crossover
                    oa, ob = self.cross_over(pa, pb)
                    skf_oa, skf_ob = skf_pa, skf_pa
                
                elif np.random.uniform() < rmp:
                    # Inter crossover
                    oa, ob = self.cross_over(pa, pb)
                    skf_oa, skf_ob = skf_pa, skf_pa
                
                else:
                    # Intra crossover
                    # select pa' and pb'
                    idx_pa2 = np.random.choice(np.where(skill_factor_arr == skf_pa)[0])
                    while idx_pa2 == idx_pa:
                        idx_pa2 = np.random.choice(np.where(skill_factor_arr == skf_pa)[0])
                    idx_pb2 = np.random.choice(np.where(skill_factor_arr == skf_pb)[0])
                    while idx_pb2 == idx_pb:
                        idx_pb2 = np.random.choice(np.where(skill_factor_arr == skf_pb)[0])

                    pa2 = population[idx_pa2]
                    pb2 = population[idx_pb2]

                    oa, _ = self.cross_over(pa, pa2)
                    ob, _ = self.cross_over(pb, pb2)

                    skf_oa, skf_ob = skf_pa, skf_pb
                
                # mutation
                oa = self.mutation(oa)
                ob = self.mutation(ob)

                offspring = np.append(offspring, [oa, ob], axis = 0)
                offspring_skill_factor = np.append(offspring_skill_factor, [skf_oa, skf_ob], axis = 0)
            
            offspring_fcost = factorial_cost(offspring, offspring_skill_factor, tasks)

            # merge
            population = np.append(population, offspring, axis = 0)
            skill_factor_arr = np.append(skill_factor_arr, offspring_skill_factor, axis = 0)
            pop_fcost = np.append(pop_fcost, offspring_fcost, axis = 0)

            # selection
            pop_fitness = 1/factorial_rank(pop_fcost, skill_factor_arr, len(tasks))
            idx = self.selection(skill_factor_arr, pop_fitness, [num_inds_each_task] * len(tasks))

            population = population[idx]
            skill_factor_arr = skill_factor_arr[idx]
            pop_fcost = pop_fcost[idx]

            #save history
            self.history_cost = np.append(self.history_cost, 
                [[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))]], 
                axis = 0
            )

            #print
            if (epoch + 1) % (num_generations // num_epochs_printed) == 0:
                if log_oneline == True:
                    sys.stdout.write('\r')
                sys.stdout.write('Epoch [{}/{}], [%-20s] %3d%% ,func_val: {}'
                    .format(epoch + 1, num_generations,[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))])
                    % ('=' * ((epoch + 1) // (num_generations // 20)) + '>' , (epoch + 1) * 100 // num_generations)
                    )
                if log_oneline == False:
                    print("\n")
                sys.stdout.flush()
        print('End')

        #solve
        sol_idx = [np.argmin(pop_fcost[np.where(skill_factor_arr == idx)]) for idx in range (len(tasks))]
        self.solve = [task.decode(population[np.where(skill_factor_arr == idx)][sol_idx[idx]]) for idx, task in enumerate(tasks)]

        return self.solve, self.history_cost            

