import numpy as np

class fnc:
    limited_space = False
    upper_bound = None
    lower_bound = None
    name = ''

    def __init__(self, d, shift = 0,limited_space: bool = False, lower_bound = None, upper_bound = None):
        self.d = d
        self.name = self.__class__.__name__ + "_d:"+ str(d) + "_translation:" + str(shift)
        if type(shift) != list:
            self.shift = shift
        else:
            assert d % len(shift) == 0
            self.shift = np.array([[i] * int(d / len(shift)) for i in shift ]).reshape(-1, )
        
        if limited_space == True:
            if lower_bound is not None and upper_bound is not None:
                self.limited_space = limited_space
                self.lower_bound = lower_bound
                self.upper_bound = upper_bound
            else:
                raise Exception("Limitted spacce must have both inf and sup val")

    def encode(self, x):
        x_encode = x
        x_encode -= 2*self.shift
        if self.limited_space == True:
            x_encode = (x_encode - self.lower_bound)/(self.upper_bound - self.lower_bound)
        return x_encode

    def decode(self, x):
        x_decoded = x[:self.d]
        if self.limited_space == True:
            x_decoded = x_decoded * (self.upper_bound - self.lower_bound) + self.lower_bound
        x_decoded = x_decoded + self.shift
        return x_decoded

class sphere(fnc):
    '''
    global optima = 0^d
    '''
    def func(self, x):
        x = self.decode(x)
        
        return np.sum(x**2, axis = 0)

class weierstrass(fnc):
    '''
    global optima = 0^d
    '''
    a = 0.5
    b = 3
    k_max = 21
    def func(self, x):
        x = self.decode(x)

        left = 0
        for i in range(self.d):
            left += np.sum(self.a ** np.arange(self.k_max) * \
                np.cos(2*np.pi * self.b ** np.arange(self.k_max) * (x[i]  + 0.5)))
            
        right = self.d * np.sum(self.a ** np.arange(self.k_max) * \
            np.cos(2 * np.pi * self.b ** np.arange(self.k_max) * 0.5)
        )
        return left - right

class ackley(fnc):
    '''
    global optima = 0^d
    '''
    a = 20
    b = 0.2
    c = 2*np.pi 
        
    def func(self, x):
        x = self.decode(x)
        
        return (-self.a * np.exp(-self.b*np.sqrt(np.average(x**2)))
        - np.exp(np.average(np.cos(self.c * x)))
        + self.a
        + np.exp(1)
        )
class rosenbrock(fnc):
    '''
    global optima = 1^d
    '''

    def func(self, x):
        x = self.decode(x)
        
        l = 100*np.sum((np.delete(x, 0, 0) - np.delete(x, -1, 0 )**2) ** 2)
        r = np.sum((np.delete(x, -1, 0) - 1) ** 2)
        return l + r
class schwefel(fnc):
    '''
    global optima = [420.9687]^d
    '''

    def func(self, x):
        x = self.decode(x)
        
        return 418.9829*self.d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
class griewank(fnc):
    ''' 
    global optima = [0] ^ d
    '''

    def func(self, x):
        x = self.decode(x)

        return np.sum(x**2) / 4000 \
            - np.prod(np.cos(x / np.sqrt((np.arange(self.d) + 1)))) + 1        
class rastrigin(fnc):
    ''' 
    global optima = 0 ^ d
    '''

    def func(self, x):
        x = self.decode(x)
        
        return 10 * self.d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
