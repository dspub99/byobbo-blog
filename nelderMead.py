# I ran this from a cell in a Jupyter notebook.
#

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

class Parabola:
    def __init__(self):
        # Express (x-x0)**2 as a*(x**2) + b*x + c
        # The maximum value of q is q=0.
        x0 = .314
        self.a = 1
        self.b = -2 * x0
        self.c = x0**2
        
        self.x = []
        self.q = []
        

    def __call__(self, x):
        q = self.a*(x**2) + self.b*x + self.c
        self.x.append(x)
        self.q.append(q)
        return q
    
pnm = Parabola()
x = minimize(pnm, x0=(np.random.uniform(-1,1),), method='Nelder-Mead').x
plt.plot(-np.array(pnm.q), '.--')
plt.xlabel('experiments')
plt.ylabel('q(x)')
plt.show();
