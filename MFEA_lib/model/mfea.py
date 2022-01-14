from typing import Tuple
import numpy as np
from ..operators import CrossOver, Mutation, Selection
from ..tasks.function import AbstractFunc
from ..GA import population_init, factorial_cost, factorial_rank, skill_factor_best_task
import sys
import matplotlib.pyplot as plt

class AbstractModel():

    def __init__(self, seed = None) -> None:
        self.history_cost: np.ndarray
        self.solve: list[np.ndarray]
        if seed is None:
            pass
        else:
            np.random.seed(seed)
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

                if skf_pa == skf_pb:
                    # intra - crossover
                    oa, ob = self.cross_over(pa, pb, type = 'intra')
                    skf_oa, skf_ob = np.random.choice([skf_pa, skf_pb], size= 2, replace= True)
                elif np.random.uniform() < rmp:
                    # inter - crossover
                    oa, ob = self.cross_over(pa, pb, type = 'inter')
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

            while len(offspring) < num_inds_each_task * len(tasks):
                [idx_pa, idx_pb] = np.random.choice(len(population), size= 2, replace= False)
                [pa, pb], [skf_pa, skf_pb] = population[[idx_pa, idx_pb]], skill_factor_arr[[idx_pa, idx_pb]]

                if skf_pa == skf_pb:
                    # Intra crossover
                    oa, ob = self.cross_over(pa, pb, type = 'intra')
                    skf_oa, skf_ob = skf_pa, skf_pa
                
                elif np.random.uniform() < rmp:
                    # Inter crossover
                    oa, ob = self.cross_over(pa, pb, type = 'inter')
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

                    oa, _ = self.cross_over(pa, pa2, type = 'inter')
                    ob, _ = self.cross_over(pb, pb2, type = 'inter')

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

class SA_MFEA(AbstractModel):
    def renderRMP(self, title = None, figsize = None, dpi = 200):
        if figsize is None:
            figsize = (30, 30)
        if title is None:
            title = self.__class__.__name__
        fig = plt.figure(figsize= figsize, dpi = dpi)
        fig.suptitle(title, size = 15)
        fig.set_facecolor("white")

        for i in range (len(self.tasks)):
            for j in range (len(self.tasks)):
                r, c = min(i, j), max(i, j)
                if i != j: 
                    plt.subplot(int(np.ceil(len(self.tasks) / 3)), 3, i + 1)
                    plt.plot(np.arange(len(self.saved_rmp[r][c])), np.array(self.saved_rmp[r][c])[:, 0], label= 'task: ' +str(j + 1))
                    plt.legend()
                else:
                    plt.subplot(int(np.ceil(len(self.tasks) / 3)), 3, i + 1)
                    plt.plot(np.arange(1000), np.ones_like(np.arange(1000)), label= 'task: ' +str(j + 1))
                    plt.legend()
            plt.title('task ' + str( i + 1))
            plt.xlabel("Epoch")
            plt.ylabel("M_rmp")
            plt.ylim(bottom = -0.1, top = 1.1)

        return fig

    def success_history_memory_update(self, memory_M:np.ndarray, next_pos: np.ndarray, S: list, delta: list):
        for i in range(len(self.tasks)):
            for j in range(i+ 1, len(self.tasks)):
                if len(S[i][j]) != 0:
                    memory_M[i, j][next_pos[i][j]] =\
                        np.sum(np.array(delta[i][j]) * np.array(S[i][j])**2)/np.sum(np.array(delta[i][j]) * (np.array(S[i][j]))+ 1e-10)
                    next_pos[i, j] = (next_pos[i, j] + 1) % memory_M.shape[2]
        return memory_M, next_pos

    def compile(self, cross_over = CrossOver.SBX_CrossOver(), mutation=  Mutation.Polynomial_Mutation(), selection= Selection.ElitismSelection()):
        super().compile(cross_over, mutation, selection)
    
    def fit(self, tasks: list[AbstractFunc], MAXEVALS, num_inds_each_task=100, nb_inds_min = None, H = 30, sigmoid = 0.1,
                evaluate_initial_skillFactor=True, range_init_pop= [0, 1]) -> tuple[list[np.ndarray], np.ndarray]:
        # LSA or SA
        if nb_inds_min is not None:
            assert num_inds_each_task >= nb_inds_min
        else:
            nb_inds_min = num_inds_each_task
            
        self.tasks = tasks

        # initial history of factorial cost -> for render
        self.history_cost = np.empty((0, len(tasks)), np.float) 

        # dim of Unified search space
        dim_uss = max([t.dim for t in tasks])

        # initial population
        inf, sup = range_init_pop
        population, skill_factor_arr = population_init(
            N = num_inds_each_task,
            num_tasks= len(tasks),
            d = dim_uss,
            min_val= inf,
            max_val= sup
        )

        if evaluate_initial_skillFactor:
            skill_factor_arr = skill_factor_best_task(population, tasks)
        pop_fcost = factorial_cost(population, skill_factor_arr, tasks)
        pop_fitness = 1/factorial_rank(pop_fcost, skill_factor_arr, len(tasks))

        # SA params:
        eval_k = np.zeros(len(tasks))
        max_Eval = int(MAXEVALS / len(tasks))
        epoch = 0

        # Initial success hitory memory M
        M_rmp = np.ones((len(tasks), len(tasks), H))/2
        next_update_pos_M = np.zeros((len(tasks), len(tasks)), np.int)

        # mean and std of rmp each generations
        # len(tasks) * len(tasks) * generations * 2
        self.saved_rmp = np.array([[[[0.5, 0]]]* len(tasks)] * len(tasks)).tolist()

        # save history_cost
        self.history_cost = np.append(self.history_cost, 
            [[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))]], 
            axis = 0
        )
        epoch = 1

        # hisory rmp of this generation
        rmp_this_gen = np.empty((len(tasks), len(tasks), 0)).tolist()

        while np.sum(eval_k) <= MAXEVALS:
            S = np.empty((len(tasks), len(tasks), 0)).tolist()
            delta = np.empty((len(tasks), len(tasks), 0)).tolist()

            # initial offspring of generation
            offspring = np.empty((0, dim_uss))
            offspring_skill_factor = np.empty((0, ), np.int)
            offspring_fcost = np.empty((0, ))

            while len(offspring) < len(population):
                [idx_pa, idx_pb] = np.random.choice(len(population), size= 2, replace= False)
                [pa, pb], [skf_pa, skf_pb] = population[[idx_pa, idx_pb]], skill_factor_arr[[idx_pa, idx_pb]]

                if np.sum(eval_k) >= epoch * num_inds_each_task * len(tasks):
                    #save history
                    self.history_cost = np.append(self.history_cost, 
                        [[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))]], 
                        axis = 0
                    ) 

                    # save mean and std rmp
                    for i in range(len(tasks)):
                        for j in range(i + 1, len(tasks)):
                            if len(rmp_this_gen[i][j]) != 0:
                                mu = np.mean(rmp_this_gen[i][j])
                                std = np.std(rmp_this_gen[i][j])
                                self.saved_rmp[i][j].append([mu, std])

                    # hisory rmp of this generation
                    rmp_this_gen = np.empty((len(tasks), len(tasks), 0)).tolist()

                    #print
                    sys.stdout.write('\r')
                    sys.stdout.write('Epoch {}, [%-20s] %3d%% ,pop_size: {}, func_val: {}'
                        .format(epoch, len(population),[np.min(
                            np.append(pop_fcost, offspring_fcost)[np.where(np.append(skill_factor_arr, offspring_skill_factor) == idx)[0]]
                            ) for idx in range (len(tasks))])
                        % ('=' * np.int((np.sum(eval_k) + 1) // (MAXEVALS // 20)) + '>' , (np.sum(eval_k) + 1) * 100 // MAXEVALS)
                        )
                    sys.stdout.flush()

                    epoch += 1


                if skf_pa == skf_pb:
                    # Intra-crossover + mutate
                    oa, ob = self.cross_over(pa, pb, type= 'intra')
                    oa, ob = self.mutation(oa), self.mutation(ob)
                    skf_oa, skf_ob = skf_pa, skf_pa

                    # Evaluate oa, ob
                    fcost_oa, fcost_ob = tasks[skf_oa].func(oa), tasks[skf_ob].func(ob)
                    eval_k[skf_oa] += 1
                    eval_k[skf_ob] += 1
                
                else:
                    # swap
                    if skf_pa > skf_pb:
                        pa, pb = pb, pa
                        skf_pa, skf_pb = skf_pb, skf_pa

                    # get rmp
                    mu = np.random.choice(M_rmp[skf_pa, skf_pb])
                    rmp = -1
                    while rmp <= 0:
                        rmp = np.random.normal(mu, sigmoid)
                    if rmp > 1: rmp = 1

                    # save generation's rmp
                    rmp_this_gen[skf_pa][skf_pb].append(rmp)

                    # Inter-TaskCrossover
                    if np.random.uniform() < rmp:
                        oa, ob = self.cross_over(pa, pb, type= 'inter')
                        oa, ob = self.mutation(oa), self.mutation(ob)
                        skf_oa, skf_ob = np.random.choice([skf_pa, skf_pb], size= 2, replace= True)
                        
                        # Evaluate oa, ob
                        fcost_oa, fcost_ob = tasks[skf_oa].func(oa), tasks[skf_ob].func(ob)
                        eval_k[skf_oa] += 1
                        eval_k[skf_ob] += 1
                    else:
                        # select pa' and pb'
                        idx_pa2 = np.random.choice(np.where(skill_factor_arr == skf_pa)[0])
                        while idx_pa2 == idx_pa:
                            idx_pa2 = np.random.choice(np.where(skill_factor_arr == skf_pa)[0])
                        idx_pb2 = np.random.choice(np.where(skill_factor_arr == skf_pb)[0])
                        while idx_pb2 == idx_pb:
                            idx_pb2 = np.random.choice(np.where(skill_factor_arr == skf_pb)[0])

                        pa2 = population[idx_pa2]
                        pb2 = population[idx_pb2]

                        oa, _ = self.cross_over(pa, pa2, type= 'intra')
                        ob, _ = self.cross_over(pb, pb2, type= 'intra')                      
                        oa, ob = self.mutation(oa), self.mutation(ob)
        
                        skf_oa, skf_ob = skf_pa, skf_pb

                        # Evaluate oa, ob
                        fcost_oa, fcost_ob = tasks[skf_oa].func(oa), tasks[skf_ob].func(ob)
                        eval_k[skf_oa] += 1
                        eval_k[skf_ob] += 1

                    
                    # Calculate the maximum improvement percetage
                    Delta = 0
                    if skf_oa == skf_pa:
                        Delta = max(Delta, 
                            (pop_fcost[idx_pa] - fcost_oa)/(pop_fcost[idx_pa] + 1e-100)
                        )
                    else:
                        Delta = max(Delta, 
                            (pop_fcost[idx_pb] - fcost_oa)/(pop_fcost[idx_pb]+ 1e-100)
                        )
                    if skf_ob == skf_pa:
                        Delta = max(Delta, 
                            (pop_fcost[idx_pa] - fcost_ob)/(pop_fcost[idx_pa]+ 1e-100)
                        )
                    else:
                        Delta = max(Delta, 
                            (pop_fcost[idx_pb] - fcost_ob)/(pop_fcost[idx_pb]+ 1e-100)
                        )
                    
                    if Delta > 0:
                        S[skf_pa][skf_pb].append(rmp)
                        delta[skf_pa][skf_pb].append(Delta)
                
                offspring = np.append(offspring, [oa, ob], axis = 0)
                offspring_skill_factor = np.append(offspring_skill_factor, [skf_oa, skf_ob], axis = 0)
                offspring_fcost = np.append(offspring_fcost, [fcost_oa, fcost_ob], axis = 0)

            # update succes history memory 
            M_rmp, next_update_pos_M = self.success_history_memory_update(M_rmp, next_update_pos_M, S, delta)

            # merge
            population = np.append(population, offspring, axis = 0)
            skill_factor_arr = np.append(skill_factor_arr, offspring_skill_factor, axis = 0)
            pop_fcost = np.append(pop_fcost, offspring_fcost, axis = 0)

            # selection
            pop_fitness = 1/factorial_rank(pop_fcost, skill_factor_arr, len(tasks))
            nb_inds_tasks = min(((nb_inds_min - num_inds_each_task)/max_Eval * eval_k + num_inds_each_task).tolist(), [num_inds_each_task] * len(tasks))
            idx = self.selection(skill_factor_arr, pop_fitness, nb_inds_tasks= nb_inds_tasks)

            population = population[idx]
            skill_factor_arr = skill_factor_arr[idx]
            pop_fcost = pop_fcost[idx]

        #solve
        sol_idx = [np.argmin(pop_fcost[np.where(skill_factor_arr == idx)]) for idx in range (len(tasks))]
        self.solve = [task.decode(population[np.where(skill_factor_arr == idx)][sol_idx[idx]]) for idx, task in enumerate(tasks)]

        return self.solve, self.history_cost
