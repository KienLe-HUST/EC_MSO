# %%
from func import sphere, weierstrass, ackley, rosenbrock, schwefel, griewank, rastrigin
import pandas as pd
from scipy.io import loadmat


# %%
ci_h = loadmat("../references/CEC17/Tasks/CI_H.mat")
ci_m = loadmat("../references/CEC17/Tasks/CI_M.mat")
ci_l = loadmat("../references/CEC17/Tasks/CI_L.mat")
pi_h = loadmat("../references/CEC17/Tasks/PI_H.mat")
pi_m = loadmat("../references/CEC17/Tasks/PI_M.mat")
pi_l = loadmat("../references/CEC17/Tasks/PI_L.mat")
ni_h = loadmat("../references/CEC17/Tasks/NI_H.mat")
ni_m = loadmat("../references/CEC17/Tasks/NI_M.mat")
ni_l = loadmat("../references/CEC17/Tasks/NI_L.mat")



# %%
pi_h.get("GO_Task2")

# %%
def CEC17_benchmark_10tasks():
    tasks = [
    sphere(     50,shift= 0,    limited_space= True, lower_bound= -100, upper_bound= 100),   # 0
    sphere(     50,shift= 80,   limited_space= True, lower_bound= -100, upper_bound= 100),  # 80
    sphere(     50,shift= -80,  limited_space= True, lower_bound= -100, upper_bound= 100), # -80
    weierstrass(25,shift= -0.4, limited_space= True, lower_bound= -0.5, upper_bound= 0.5), # -0.4
    rosenbrock( 50,shift= -1,   limited_space= True, lower_bound= -50, upper_bound= 50),# 0
    ackley(     50,shift= 40,   limited_space= True, lower_bound= -50, upper_bound= 50),    # 40
    weierstrass(50,shift= -0.4, limited_space= True, lower_bound= -0.5, upper_bound= 0.5), # -0.4
    schwefel(   50,shift= 0,    limited_space= True, lower_bound= -500, upper_bound= 500), # 420.9687
    griewank(   50,shift= [-80, 80],limited_space= True, lower_bound= -100, upper_bound= 100), # -80, 80
    rastrigin(  50,shift= [-40, 40],limited_space= True, lower_bound= -50, upper_bound= 50),# -40, 40
    ]
    return tasks
    

# %%
def GECCO20_benchmark_50tasks()


