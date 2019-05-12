
import numpy as np


# The "real" mountain car.
# Ok, it's still a simulation.  But it's not
#  the simulation used for training.
#
# In this sim, we've fixed shapeA and shapeB,
#  modified the range of allowed forces and velocitiy,
#  moved the goal, and added drag.
#

class RealMountainCar():
    def __init__(self):
        self._shapeA = .0015
        self._shapeB = 4.5
        self._min_force = -1.2 # -1.0
        self._max_force = .9 # 1.0
        self._min_x = -1.2
        self._max_x = 0.6
        self._max_vel = .1 # 0.07
        self._goal_x = 0.55 #0.45
        self._power = 0.0015
        self._drag = 1 / self._max_vel
        self._forceRange = (self._min_force, self._max_force)
        self._stopped = False

        # Start at the bottom of the hill, perfectly still.
        # Why?  Because it's a car.
        x0 = -np.pi/2/self._shapeB
        v0 = 0
        self._state = np.array([x0, v0])

    def forceRange(self):
        return self._forceRange

    def state(self):
        return self._state

    def stopped(self):
        return self._stopped
    
    def step(self, force0):
        if self._stopped:
            return 0
        
        x = self._state[0]
        v = self._state[1]
        force = min(max(force0, self._min_force), self._max_force)

        # grad
        force -= self._drag * v**2
        
        v += force*self._power - self._shapeA * np.cos(self._shapeB*x)
        if (v > self._max_vel):
            v = self._max_vel
        if (v < -self._max_vel):
            v = -self._max_vel

            
        x += v
        if (x > self._max_x):
            x = self._max_x
        if (x < self._min_x):
            x = self._min_x
        if (x==self._min_x and v<0):
            v = 0

        if x >= self._goal_x:
            reward = 100.0
            self._stopped = True
        else:
            reward = 0

        reward-= .1 * (force0**2)

        self._state[0] = x
        self._state[1] = v

        return reward
    
def test_still():
    mmc = RealMountainCar()
    for _ in range(1000):
        v = mmc.state()[1]
        assert( np.abs(v) < 1e-6 )
        mmc.step(0)

def test_mmc():
    mmc = RealMountainCar()
    rng = np.random.RandomState(13)
    fa, fb = mmc.forceRange()

    rTot = 0
    for _ in range(10):
        f = 1 # rng.uniform(fa, fb)
        rTot += mmc.step(f)
        x,v = mmc.state()
        print (":", rTot, x, v)
        
                   
