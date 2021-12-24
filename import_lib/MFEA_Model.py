from typing import Tuple
import numpy as np
import sys
from import_lib.GA import population_init, factorial_cost, factorial_rank, skill_factor_best_task, polynomial_mutation, sbx_crossover

def MFEA1(num_epochs, num_inds_each_task, range_init_pop = [0, 1],tasks = [], rmp = 0.1, nc = 15, nm = 15, rm = 0.02,
                one_line = False, num_epochs_printed = 20, polynomial_all_gen = False, evaluate_initial_skillFactor = True) -> Tuple[list, np.ndarray]:
    
    #save history of factorial cost
    history_cost = np.empty((0, len(tasks)), np.float)
    
    max_d = 0
    for t in tasks:
        if max_d < t.d:
            max_d = t.d

    # initial population and skill_factor_arr
    inf, sup = range_init_pop
    population, skill_factor_arr = population_init(
        N = num_inds_each_task, 
        num_tasks = len(tasks), 
        d = max_d,
        min_val = inf,
        max_val = sup,
    )
    if evaluate_initial_skillFactor:
        skill_factor_arr = skill_factor_best_task(population, tasks)

    pop_fcost = factorial_cost(population, skill_factor_arr, tasks)

    history_cost = np.append(history_cost, [[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))]], axis = 0)
    
    for epoch in range(num_epochs):

        #initial offspring for epoch
        offspring = np.empty((0, max_d))
        offspring_skill_factor = np.empty((0, 1), np.int)
        offspring_fcost = np.empty((0, 1))

        while len(offspring) < len(population):
            [idx_a, idx_b] = np.random.randint(len(population), size = 2)
            [pa, pb], [skf_a, skf_b] = population[[idx_a, idx_b]], skill_factor_arr[[idx_a, idx_b]]

            if skf_a == skf_b:
                # intra - crossover
                ca, cb = sbx_crossover(pa, pb, nc)
                            
                ca = polynomial_mutation(ca,nm, rm, polynomial_all_gen)
                cb = polynomial_mutation(cb,nm, rm, polynomial_all_gen)   
                
                offspring = np.append(offspring, [ca, cb], axis = 0)
                offspring_skill_factor = np.append(offspring_skill_factor, [skf_a, skf_a])

            elif np.random.uniform() <= rmp:
                # inter - crossover
                ca, cb = sbx_crossover(pa, pb, nc)
                            
                ca = polynomial_mutation(ca,nm, rm, polynomial_all_gen)
                cb = polynomial_mutation(cb,nm, rm, polynomial_all_gen)   

                offspring = np.append(offspring, [ca, cb], axis = 0)
                skf_ca, skf_cb = np.random.choice([skf_a, skf_b], 2, True)
                offspring_skill_factor = np.append(offspring_skill_factor, [skf_ca, skf_cb])
   
            else:
                # select pa' and pb'
                idx_pa2 = np.random.choice(np.where(skill_factor_arr == skf_a)[0])
                idx_pb2 = np.random.choice(np.where(skill_factor_arr == skf_b)[0])
                pa2 = population[idx_pa2]
                pb2 = population[idx_pb2]

                ca, _ = sbx_crossover(pa, pa2)
                cb, _ = sbx_crossover(pb, pb2)
                            
                ca = polynomial_mutation(ca,nm, rm, polynomial_all_gen)
                cb = polynomial_mutation(cb,nm, rm, polynomial_all_gen)   
            
                offspring = np.append(offspring, [ca, cb], axis = 0)
                offspring_skill_factor = np.append(offspring_skill_factor, [skf_a, skf_b])   
    
        # merge
        offspring_fcost = factorial_cost(offspring, offspring_skill_factor, tasks)

        population = np.append(population, offspring, axis = 0)
        skill_factor_arr = np.append(skill_factor_arr, offspring_skill_factor, axis = 0)
        pop_fcost = np.append(pop_fcost, offspring_fcost, axis = 0)

        # selection
        pop_finess = 1/factorial_rank(pop_fcost, skill_factor_arr, len(tasks))
        num_inds_pop = np.int(num_inds_each_task)* len(tasks)
        selected_inds = np.argsort(-pop_finess)[:num_inds_pop] 
        population, skill_factor_arr = population[selected_inds], skill_factor_arr[selected_inds]
        pop_fcost = pop_fcost[selected_inds]

        #save history
        history_cost = np.append(history_cost, [[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))]], axis = 0)

        #print
        if (epoch + 1) % (num_epochs // num_epochs_printed) == 0:
            if one_line == True:
                sys.stdout.write('\r')
            sys.stdout.write('Epoch [{}/{}], [%-20s] %3d%% ,func_val: {}'
                .format(epoch + 1, num_epochs,[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))])
                % ('=' * ((epoch + 1) // (num_epochs // 20)) + '>' , (epoch + 1) * 100 // num_epochs)
                )
            if one_line == False:
                print("\n")
            sys.stdout.flush()
    print('END!')
    #find solve
    sol_idx = [np.argmin(pop_fcost[np.where(skill_factor_arr == idx)]) for idx in range (len(tasks))]
    sol = [task.decode(population[np.where(skill_factor_arr == idx)][sol_idx[idx]]) for idx, task in enumerate(tasks)]

    return sol, history_cost

def MFEA_base(num_epochs, num_inds_each_task, range_init_pop = [0, 1],tasks = [], rmp = 0.1, nc = 15, nm = 15, rm = 0.02,
                one_line = False, num_epochs_printed = 20, polynomial_all_gen = False, evaluate_initial_skillFactor = True) -> Tuple[list, np.ndarray]:
    
    #save history of factorial cost
    history_cost = np.empty((0, len(tasks)), np.float)
    
    max_d = 0
    for t in tasks:
        if max_d < t.d:
            max_d = t.d

    # initial population and skill_factor_arr
    inf, sup = range_init_pop
    population, skill_factor_arr = population_init(
        N = num_inds_each_task, 
        num_tasks = len(tasks), 
        d = max_d,
        min_val = inf,
        max_val = sup,
    )
    if evaluate_initial_skillFactor:
        skill_factor_arr = skill_factor_best_task(population, tasks)
        
    pop_fcost = factorial_cost(population, skill_factor_arr, tasks)

    history_cost = np.append(history_cost, [[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))]], axis = 0)
    
    for epoch in range(num_epochs):

        #initial offspring for epoch
        offspring = np.empty((0, max_d))
        offspring_skill_factor = np.empty((0, 1), np.int)
        offspring_fcost = np.empty((0, 1))

        while len(offspring) < len(population):
            [idx_a, idx_b] = np.random.randint(len(population), size = 2)
            [pa, pb], [skf_a, skf_b] = population[[idx_a, idx_b]], skill_factor_arr[[idx_a, idx_b]]

            if skf_a == skf_b:
                # intra - crossover
                ca, cb = sbx_crossover(pa, pb, nc)
                
                offspring = np.append(offspring, [ca, cb], axis = 0)
                offspring_skill_factor = np.append(offspring_skill_factor, [skf_a, skf_a])

            elif np.random.uniform() <= rmp:
                # inter - crossover
                ca, cb = sbx_crossover(pa, pb, nc)

                offspring = np.append(offspring, [ca, cb], axis = 0)
                skf_ca, skf_cb = np.random.choice([skf_a, skf_b], 2, True)
                offspring_skill_factor = np.append(offspring_skill_factor, [skf_ca, skf_cb])
   
            else:
                # mutation  
                ca = polynomial_mutation(pa,nm, rm, polynomial_all_gen)
                cb = polynomial_mutation(pb,nm, rm, polynomial_all_gen)   
            
                offspring = np.append(offspring, [ca, cb], axis = 0)
                offspring_skill_factor = np.append(offspring_skill_factor, [skf_a, skf_b])   
    
        # merge
        offspring_fcost = factorial_cost(offspring, offspring_skill_factor, tasks)

        population = np.append(population, offspring, axis = 0)
        skill_factor_arr = np.append(skill_factor_arr, offspring_skill_factor, axis = 0)
        pop_fcost = np.append(pop_fcost, offspring_fcost, axis = 0)

        # selection
        pop_finess = 1/factorial_rank(pop_fcost, skill_factor_arr, len(tasks))
        num_inds_pop = np.int(num_inds_each_task)* len(tasks)
        selected_inds = np.argsort(-pop_finess)[:num_inds_pop] 
        population, skill_factor_arr = population[selected_inds], skill_factor_arr[selected_inds]
        pop_fcost = pop_fcost[selected_inds]

        #save history
        history_cost = np.append(history_cost, [[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))]], axis = 0)

        #print
        if (epoch + 1) % (num_epochs // num_epochs_printed) == 0:
            if one_line == True:
                sys.stdout.write('\r')
            sys.stdout.write('Epoch [{}/{}], [%-20s] %3d%% ,func_val: {}'
                .format(epoch + 1, num_epochs,[np.min(pop_fcost[np.where(skill_factor_arr == idx)[0]]) for idx in range (len(tasks))])
                % ('=' * ((epoch + 1) // (num_epochs // 20)) + '>' , (epoch + 1) * 100 // num_epochs)
                )
            if one_line == False:
                print("\n")
            sys.stdout.flush()
    print('END!')
    #find solve
    sol_idx = [np.argmin(pop_fcost[np.where(skill_factor_arr == idx)]) for idx in range (len(tasks))]
    sol = [task.decode(population[np.where(skill_factor_arr == idx)][sol_idx[idx]]) for idx, task in enumerate(tasks)]

    return sol, history_cost