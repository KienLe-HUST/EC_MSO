from numpy.random.mtrand import choice
from numpy import loadtxt
from .func import sphere, weierstrass, ackley, rosenbrock, schwefel, griewank, rastrigin
import pandas as pd
from scipy.io import loadmat
import os

path = os.path.dirname(os.path.realpath(__file__))

class GECCO20_benchmark_50tasks():
    tasks_size = 50
    dim = 50

    def get_choice_function(self, ID):
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

    def get_items(self, ID):
        choice_functions = self.get_choice_function(ID)

        tasks = []

        for task_id in range(self.task_size):
            func_id = choice_functions[task_id % len(choice_functions)]
            file_dir = path + "/references/GECCO20/Tasks/benchmark_" + str(ID)
            shift_file = "/bias_" + str(task_id + 1)
            rotation_file = "/matrix_" + str(task_id + 1)
            matrix = loadtxt(file_dir + rotation_file)
            shift = loadtxt(file_dir + shift_file)

            if func_id == 1:
                tasks.append(
                    sphere(self.dim, shift= shift, rotation_matrix= matrix,
                    limited_space= True, lower_bound= -100, upper_bound= 100)
                )
            elif func_id == 2:
                tasks.append(
                    rosenbrock(self.dim, shift= shift, rotation_matrix= matrix,
                    limited_space= True, lower_bound= -50, upper_bound= 50)
                )
            elif func_id == 2:
                tasks.append(
                    rosenbrock(self.dim, shift= shift, rotation_matrix= matrix,
                    limited_space= True, lower_bound= -50, upper_bound= 50)
                )
            elif func_id == 3:
                tasks.append(
                    ackley(self.dim, shift= shift, rotation_matrix= matrix,
                    limited_space= True, lower_bound= -50, upper_bound= 50)
                )
            elif func_id == 4:
                tasks.append(
                    rastrigin(self.dim, shift= shift, rotation_matrix= matrix,
                    limited_space= True, lower_bound= -50, upper_bound= 50)
                )
            elif func_id == 5:
                tasks.append(
                    griewank(self.dim, shift= shift, rotation_matrix= matrix,
                    limited_space= True, lower_bound= -100, upper_bound= 100)
                )
            elif func_id == 6:
                tasks.append(
                    rosenbrock(self.dim, shift= shift, rotation_matrix= matrix,
                    limited_space= True, lower_bound= -0.5, upper_bound= 0.5)
                )
            elif func_id == 7:
                tasks.append(
                    rosenbrock(self.dim, shift= shift, rotation_matrix= matrix,
                    limited_space= True, lower_bound= -500, upper_bound= 500)
                )
        return tasks    

class CEC17_benchmark():
    def get_10tasks_benchmark(self):
        tasks = [
        sphere(     50,shift= 0,    limited_space= True, lower_bound= -100, upper_bound= 100),   # 0
        sphere(     50,shift= 80,   limited_space= True, lower_bound= -100, upper_bound= 100),  # 80
        sphere(     50,shift= -80,  limited_space= True, lower_bound= -100, upper_bound= 100), # -80
        weierstrass(25,shift= -0.4, limited_space= True, lower_bound= -0.5, upper_bound= 0.5), # -0.4
        rosenbrock( 50,shift= 0,   limited_space= True, lower_bound= -50, upper_bound= 50),# 0
        ackley(     50,shift= 40,   limited_space= True, lower_bound= -50, upper_bound= 50),    # 40
        weierstrass(50,shift= -0.4, limited_space= True, lower_bound= -0.5, upper_bound= 0.5), # -0.4
        schwefel(   50,shift= 420.9687,    limited_space= True, lower_bound= -500, upper_bound= 500), # 420.9687
        griewank(   50,shift= [-80, 80],limited_space= True, lower_bound= -100, upper_bound= 100), # -80, 80
        rastrigin(  50,shift= [40, -40],limited_space= True, lower_bound= -50, upper_bound= 50),# -40, 40
        ]
        return tasks

    def get_2tasks_benchmark(self, ID):
        #TODO
        pass