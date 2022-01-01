import numpy as np

class AbstractMutation():
    def __init__(self):
        pass
    def __call__(self, p) -> np.ndarray:
        pass

class Polynomial_Mutation(AbstractMutation):
    '''
    p in [0, 1]^n
    '''
    def __init__(self, nm = 15, mutate_all_dimensions:bool = False):
        self.nm = nm
        self.mutate_all_dimensions = mutate_all_dimensions
    
    def __call__(self, p) -> np.ndarray:
        super().__call__(p)

        ind = np.copy(p)

        if self.mutate_all_dimensions:
            u = np.random.uniform()
            if u < 0.5:
                delta_l = (2*u)**(1/(self.nm + 1)) - 1
                ind = ind + delta_l * ind
            
            else: 
                delta_r = 1 - (2*(1-u))**(1/(self.nm + 1))
                ind = ind + delta_r * (1 - ind)
        else:
            # NOTE
            for i in range(len(ind)):
                if np.random.uniform() < 1/len(ind):
                    u = np.random.uniform()
                    if u < 0.5:
                        delta_l = (2*u)**(1/(self.nm + 1)) - 1
                        ind[i] = ind[i] + delta_l * ind[i]
                    
                    else: 
                        delta_r = 1 - (2*(1-u))**(1/(self.nm + 1))
                        ind[i] = ind[i] + delta_r * (1 - ind[i])
                else:
                    continue
        return ind