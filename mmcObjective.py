
import numpy as np
from mmc import MetaMountainCar

class Controller:
    def __init__(self, x):
        self._x = np.array([2*x[0], 20*(1+x[1])/2])

    def force(self, state):
        return np.tanh(self._x.dot(state))

class Evaluator:
    def __init__(self, shapeA, shapeB, bFlip):
        self._shapeA = shapeA
        self._shapeB = shapeB
        self._bFlip = bFlip

    def __call__(self, x):
        c = Controller(x)
        mcc = MetaMountainCar(self._shapeA, self._shapeB, self._bFlip)
        phi = 0
        for _ in range(1000):
            f = c.force(mcc.state())
            phi += mcc.step(f)
        return phi

class MMCObjective:
    def __init__(self, seed, __=None):
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self.nDim = 2
        self.y0 = 0
        self.scale = 1
        
    def reset(self, ie):
        if ie is not None and self._seed is not None:
            seed = self._seed + ie
            self._rng.seed(seed)
        shapeA = self._rng.uniform(.0005, .0025)
        shapeB = self._rng.uniform(2, 6)
        bFlip = self._rng.randint(2)==0
        self._eval = Evaluator(shapeA, shapeB, bFlip)


    def f(self, x):
        return -self._eval(x)

def test_mmco():
    mmco = MMCObjective(1)
    mmco.reset(3)
    mmco.f(np.array([1,1,]))
    

def test_eval():
    e = Evaluator(0.0009791365131265105, 3, False)
    print (e([.06, .4]))
    


