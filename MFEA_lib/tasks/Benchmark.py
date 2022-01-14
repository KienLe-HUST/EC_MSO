# from numpy.random.mtrand import choice
from numpy import loadtxt
from .function import AbstractFunc, Sphere, Weierstrass, Ackley, Rosenbrock, Schwefel, Griewank, Rastrigin
# import pandas as pd
# from scipy.io import loadmat
import os

path = os.path.dirname(os.path.realpath(__file__))

# %%
# ci_h = loadmat("../references/CEC17/Tasks/CI_H.mat")
# ci_m = loadmat("../references/CEC17/Tasks/CI_M.mat")
# ci_l = loadmat("../references/CEC17/Tasks/CI_L.mat")
# pi_h = loadmat("../references/CEC17/Tasks/PI_H.mat")
# pi_m = loadmat("../references/CEC17/Tasks/PI_M.mat")
# pi_l = loadmat("../references/CEC17/Tasks/PI_L.mat")
# ni_h = loadmat("../references/CEC17/Tasks/NI_H.mat")
# ni_m = loadmat("../references/CEC17/Tasks/NI_M.mat")
# ni_l = loadmat("../references/CEC17/Tasks/NI_L.mat")



# %%
# pi_h.get("GO_Task2")




class GECCO20_benchmark_50tasks():
    task_size = 50
    dim = 50

    def get_choice_function(ID) -> list[int]:
        choice_functions = []
        if ID == 1:
            choice_functions = [1]
        elif ID == 2:
            choice_functions = [2]
        elif ID == 3:
            choice_functions = [4]
        elif ID == 4:
            choice_functions = [1, 2, 3]
        elif ID == 5:
            choice_functions = [4, 5, 6]
        elif ID == 6:
            choice_functions = [2, 5, 7]
        elif ID == 7:
            choice_functions = [3, 4, 6]
        elif ID == 8:
            choice_functions = [2, 3, 4, 5, 6]
        elif ID == 9:
            choice_functions = [2, 3, 4, 5, 6, 7]
        elif ID == 10:
            choice_functions = [3, 4, 5, 6, 7]
        else:
            raise ValueError("Invalid input: ID should be in [1,10]")
        return choice_functions

    def get_items(ID) -> list[AbstractFunc]:
        choice_functions = __class__.get_choice_function(ID)

        tasks = []

        for task_id in range(__class__.task_size):
            func_id = choice_functions[task_id % len(choice_functions)]
            file_dir = path + "/__references__/GECCO20/Tasks/benchmark_" + str(ID)
            shift_file = "/bias_" + str(task_id + 1)
            rotation_file = "/matrix_" + str(task_id + 1)
            matrix = loadtxt(file_dir + rotation_file)
            shift = loadtxt(file_dir + shift_file)

            if func_id == 1:
                tasks.append(
                    Sphere(__class__.dim, shift= shift, rotation_matrix= matrix,bound= [-100, 100])
                )
            elif func_id == 2:
                tasks.append(
                    Rosenbrock(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-50, 50])
                )
            elif func_id == 3:
                tasks.append(
                    Ackley(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-50, 50])
                )
            elif func_id == 4:
                tasks.append(
                    Rastrigin(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-50, 50])
                )
            elif func_id == 5:
                tasks.append(
                    Griewank(__class__.dim, shift= shift, rotation_matrix= matrix, bound = [-100, 100])
                )
            elif func_id == 6:
                tasks.append(
                    Rosenbrock(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-0.5, 0.5])
                )
            elif func_id == 7:
                tasks.append(
                    Rosenbrock(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-500, 500])
                )
        return tasks    

class CEC17_benchmark():
    def get_10tasks_benchmark(fix = False)-> list[AbstractFunc]:
        tasks = [
        Sphere(     50,shift= 0,    bound= [-100, 100]),   # 0
        Sphere(     50,shift= 80,   bound= [-100, 100]),  # 80
        Sphere(     50,shift= -80,  bound= [-100, 100]), # -80
        Weierstrass(25,shift= -0.4, bound= [-0.5, 0.5]), # -0.4
        Rosenbrock( 50,shift= -1,   bound= [-50, 50]),# 0
        Ackley(     50,shift= 40,   bound= [-50, 50], fixed = fix),    # 40
        Weierstrass(50,shift= -0.4, bound= [-0.5, 0.5]), # -0.4
        Schwefel(   50,shift= 0,    bound= [-500, 500],fixed = fix), # 420.9687
        Griewank(   50,shift= [-80, 80], bound= [-100, 100]), # -80, 80
        Rastrigin(  50,shift= [40, -40], bound= [-50, 50]),# -40, 40
        ]
        return tasks


    def get_2tasks_benchmark(ID)-> list[AbstractFunc]:
        #TODO
        pass

