import numpy as np

def population_init(N, num_tasks, d, min_val = 0, max_val = 1):
    '''
    pop.shape = (N * num_tasks, d)
    skill_factor_arr.shape = (N * num_tasks, )
    '''
    pop = np.array([np.random.uniform(min_val, max_val, d) for i in range(N * num_tasks)])
    skill_factor_arr = np.array([[i] * N for i in range(num_tasks)]).reshape(-1, )
    return pop, skill_factor_arr

def factorial_cost(pop, skill_factor_arr, tasks = []):
    return np.array([tasks[skill_factor].func(pop[idx_inds]) for idx_inds, skill_factor in enumerate(skill_factor_arr)])

def factorial_rank(f_cost, skill_factor_arr, num_tasks):
    res = np.zeros_like(skill_factor_arr)
    for i in range (num_tasks):
        inds = np.where(skill_factor_arr == i)
        res[inds] = np.argsort(np.argsort(f_cost[inds]))
    return res + 1

def sbx_crossover(pa, pb, nc = 15):
    '''
    pa, pb in [0, 1]^n
    '''
    u = np.random.uniform()
    if u <= 0.5:
        beta = (2*u)**(1/(nc +1))
    else:
        beta = (1/(2 * (1 - u)))**(1 / (1 + nc))
        
    c1 = 0.5*((1 + beta) * pa + (1 - beta) * pb)
    c2 = 0.5*((1 - beta) * pa + (1 + beta) * pb)

    c1, c2 = np.clip(c1, 0, 1), np.clip(c2, 0, 1)
    return c1, c2

def polynomial_mutation(ind, nm = 15, rm = 0.9, all_dimensions = False):
    '''
    pa, pb in [0, 1]^n
    '''
    p = np.copy(ind)

    if all_dimensions == True:
        u = np.random.rand()
        if u <= 0.5:
            delta_l = (2*u)**(1/(nm + 1)) - 1
            return p + delta_l * p
        
        else: 
            delta_r = 1 - (2*(1-u))**(1/(nm + 1))
            return p + delta_r * (1 - p)
    else:
        for i in range(len(p)):
            if np.random.uniform() <= rm:
                u = np.random.rand()
                if u <= 0.5:
                    delta_l = (2*u)**(1/(nm + 1)) - 1
                    p[i] = p[i] + delta_l * p[i]
                
                else: 
                    delta_r = 1 - (2*(1-u))**(1/(nm + 1))
                    p[i] = p[i] + delta_r * (1 - p[i])
            else:
                continue
        return p


def skill_factor_best_task(pop, tasks):
    population = np.copy(pop)
    maxtrix_cost = np.array([np.apply_along_axis(t.func, 1, population) for t in tasks]).T
    matrix_rank_pop = np.argsort(np.argsort(maxtrix_cost, axis = 0), axis = 0) 

    N = len(population) / len(tasks)
    count_inds = np.array([0] * len(tasks))
    skill_factor_arr = np.zeros(int((N * len(tasks)),), dtype=np.int)
    condition = False
    
    while not condition:
        idx_task = np.random.choice(np.where(count_inds < N)[0])

        idx_ind = np.argsort(matrix_rank_pop[:, idx_task])[0]

        skill_factor_arr[idx_ind] = idx_task

        matrix_rank_pop[idx_ind] = len(pop) + 1
        count_inds[idx_task] += 1

        condition = np.all(count_inds == N)

    return skill_factor_arr
