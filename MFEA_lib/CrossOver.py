from typing import Tuple
import numpy as np


class AbstractCrossOver():
    def __init__(self) -> None:
        pass
    def __call__(self, pa, pb) -> Tuple[np.ndarray, np.ndarray]:
        pass

class SBX_CrossOver(AbstractCrossOver):
    def __init__(self, nc = 15):
        self.nc = nc
    def __call__(self, pa, pb) -> Tuple[np.ndarray, np.ndarray]:
        '''
        pa, pb in [0, 1]^n
        '''
        u = np.random.uniform(size = len(pa))

        beta = np.where(u < 0.5, (2*u)**(1/(self.nc +1)), (2 * (1 - u))**(-1 / (1 + self.nc)))
            
        c1 = 0.5*((1 + beta) * pa + (1 - beta) * pb)
        c2 = 0.5*((1 - beta) * pa + (1 + beta) * pb)

        c1, c2 = np.clip(c1, 0, 1), np.clip(c2, 0, 1)
        return c1, c2

