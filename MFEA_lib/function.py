import numpy as np

class AbstractFunc():
    limited_space = False
    upper_bound = None
    lower_bound = None
    name = ''

    def __init__(self, dim, shift = 0, rotation_matrix: np.ndarray = None, 
                limited_space: bool = False, lower_bound = None, upper_bound = None):
        self.dim = dim
        #NOTE
        self.name = self.__class__.name

        if rotation_matrix is not None:
            assert np.all(np.array(rotation_matrix.shape) == dim)
            self.rotation_matrix = rotation_matrix
            self.inv_rotation_matrix = np.linalg.inv(self.rotation_matrix)
        else:
            self.rotation_matrix = np.identity(dim)
            self.inv_rotation_matrix = np.identity(dim)
        
        tmp = np.array(shift)
        assert dim % len(shift) == 0
        self.shift = np.array([[i] * int(dim / len(tmp)) for i in tmp ]).reshape(-1, )

        if limited_space == True:
            if lower_bound is not None and upper_bound is not None:
                self.limited_space = limited_space
                self.lower_bound = lower_bound
                self.upper_bound = upper_bound
            else:
                raise Exception("Limitted spacce must have both inf and sup val")

    def encode(self, x):
        '''
        encode x to [0, 1]
        '''
        x_encode = x
        x_encode = self.inv_rotation_matrix @ x_encode + self.shift
        if self.limited_space == True:
            x_encode = (x_encode - self.lower_bound)/(self.upper_bound - self.lower_bound)
        return x_encode 

    def decode(self, x):
        '''
        decode x
        '''
        x_decode = x[:self.dim]
        if self.limited_space == True:
            x_decode = x_decode * (self.upper_bound - self.lower_bound) + self.lower_bound
        x_decode = self.rotation_matrix @ (x_decode - self.shift) 
        return x_decode

class Sphere(AbstractFunc):
    '''
    global optima = 0^d
    '''
    def func(self, x):
        '''
        Request: input x is encoded
        '''
        x = self.decode(x)
        return np.sum(x**2, axis = 0)

class Weierstrass(AbstractFunc):
    '''
    global optima = 0^d
    '''
    a = 0.5
    b = 3
    k_max = 21
    def func(self, x):
        '''
        Request: input x is encoded
        '''
        x = self.decode(x)
        left = 0
        for i in range(self.d):
            left += np.sum(self.a ** np.arange(self.k_max) * \
                np.cos(2*np.pi * self.b ** np.arange(self.k_max) * (x[i]  + 0.5)))
            
        right = self.d * np.sum(self.a ** np.arange(self.k_max) * \
            np.cos(2 * np.pi * self.b ** np.arange(self.k_max) * 0.5)
        )
        return left - right

class Ackley(AbstractFunc):
    '''
    global optima = 0^d
    '''
    a = 20
    b = 0.2
    c = 2*np.pi 
    def func(self, x):
        x = self.decode(x)
        return -self.a * np.exp(-self.b*np.sqrt(np.average(x**2)))\
        - np.exp(np.average(np.cos(self.c * x)))\
        + self.a\
        + np.exp(1)

class Rosenbrock(AbstractFunc):
    '''
    global optima = 0^d
    '''
    def func(self, x):
        x = self.decode(x) + 1
        l = 100*np.sum((np.delete(x, 0, 0) - np.delete(x, -1, 0 )**2) ** 2)
        r = np.sum((np.delete(x, -1, 0) - 1) ** 2)
        return l + r

class Schwefel(AbstractFunc):
    '''
    global optima = 0^d
    '''
    def func(self, x):
        x = self.decode(x) + 420.9687
        return 418.9829*self.d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
class Griewank(AbstractFunc):
    ''' 
    global optima = [0] ^ d
    '''
    def func(self, x):
        x = self.decode(x)
        return np.sum(x**2) / 4000 \
            - np.prod(np.cos(x / np.sqrt((np.arange(self.d) + 1)))) + 1

class Rastrigin(AbstractFunc):
    ''' 
    global optima = 0 ^ d
    '''
    def func(self, x):
        x = self.decode(x)
        
        return 10 * self.d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

