from copy import copy
import numpy as np

class AbstractMutation():
    def __init__(self):
        pass
    def __call__(self, p) -> np.ndarray:
        pass
class NoMutation(AbstractMutation):
    def __call__(self, p) -> np.ndarray:
        return p
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
            pm = 1/len(ind)
            idx_mutation = np.where(np.random.rand(len(ind)) < pm)[0]
            u = np.ones_like(ind)/2
            u[idx_mutation] = np.random.rand(len(idx_mutation))

            delta = np.where(u < 0.5,
                # delta_l
                (2*u)**(1/(self.nm + 1)) - 1,
                # delta_r
                1 - (2*(1-u))**(1/(self.nm + 1))
            )

            ind = np.where(delta < 0,
                # delta_l: ind -> 0
                ind + delta * ind,
                # delta_r: ind -> 1
                ind + delta * (1 - ind)
            )

        return ind

class GaussMutation(AbstractMutation):
    '''
    p in [0, 1]^n
    '''
    def __init__(self, scale = 1):
        self.scale = scale
    
    def __call__(self, p) -> np.ndarray:   
        super().__call__(p)

        ind = np.copy(p)
        pm = 1/len(ind)

        idx_mutation = np.where(np.random.rand(len(ind)) < pm)[0]

        t = ind[idx_mutation] + np.random.normal(0, self.scale, size = len(idx_mutation))
        
        t = np.where(t > 1, ind[idx_mutation] + np.random.rand() * (1 - ind[idx_mutation]), t)
        t = np.where(t < 0, np.random.rand() * ind[idx_mutation], t)

        ind[idx_mutation] = t
        return ind

class GMDScale(AbstractMutation):
    '''
    p in [0, 1]^n
    '''
    def __init__(self, scale = .1, rate = 0.5):
        self.scale = scale
        self.rate = rate
    def __call__(self, p, scale: np.ndarray):
        super().__call__(p)

        ind = np.copy(p)
        pm = 1/len(ind)

        idx_mutation = np.where(np.random.rand(len(ind)) < pm)[0]

        t = ind[idx_mutation] + np.random.normal(0, np.array(scale * self.rate)[idx_mutation], size = len(idx_mutation))
        
        t = np.where(t > 1, ind[idx_mutation] + np.random.rand() * (1 - ind[idx_mutation]), t)
        t = np.where(t < 0, np.random.rand() * ind[idx_mutation], t)

        ind[idx_mutation] = t
        return ind