#!/usr/bin/env python

import numpy as np

class Objective:
    def __init__(self, seed, nDim, spread):
        self._seed = seed
        self._spread = spread
        self._rng = np.random.RandomState(seed)
        self.nDim = nDim
        
    def reset(self, ie):
        if ie is not None and self._seed is not None:
            seed = self._seed + ie
            self._rng.seed(seed)
        self.y0 = 100*self._spread*self._rng.uniform(-1, 1)
        self.scale = 10 ** ( self._spread*self._rng.uniform(-2,2) )
        self._x0 = self._spread*self._rng.uniform(-1, 1, size=(self.nDim,))
        
    def f(self, x):
        return self.y0 + self.scale*self.f0(x)

class Sphere(Objective):
    def __init__(self, seed, nDim, spread=1):
        super().__init__(seed, nDim, spread)

    def f0(self, x):
        return ((x-self._x0)**2).sum()

def test1():
    rng = np.random.RandomState(12)
    for n in range(10):
        Sphere(1, n)(rng.uniform(-1,1,size=(n,)))
    
