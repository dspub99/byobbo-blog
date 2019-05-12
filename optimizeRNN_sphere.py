#!/usr/bin/env python

# You can run this with:
#  mpiexec -n NPROCS optimizeRNN_sphere.py NDIM mybbo.json
# where NPROCS is the number of processes you want to spawn,
# and NDIM is the dimension of the sphere function you want
# to optimize.
#   NPROCS <= number of cores
#   NDIM in [1, 3, 10]

# For example:
#  mpiexec -n 4 optimizeRNN_sphere.py 1 mybbo.json
#


if __name__=="__main__":

    import time
    import sys

    import numpy as np
    import json_tricks

    from byobbo import BYOBBO
    from bbo import RNNTypes
    from objectives import Sphere

    nDim = int(sys.argv[1])
    assert(nDim in [1, 3, 10])
    jsonFn = sys.argv[2]

    
    nObjIter = 3 + nDim
    seed0 = None
    nEvalsPerW = 100
    sigma0 = .1
    objClass = Sphere
    RNNType = RNNTypes.Plain
    if nDim==10:
        NumHidden = [256]*2
    else:
        NumHidden = [16]*2

    rng = np.random.RandomState(seed0)
    byobbo = BYOBBO(seed=rng.randint(9999), rnnType=RNNType, nh=NumHidden, nObjIter=nObjIter, nReport=1,
                    sigma0=sigma0, nPop = None, tMax=None, nEvalsPerW=nEvalsPerW)
    

    objFit = objClass(rng.randint(9999), nDim)
    nIter = 0
    oyff = 1e9
    yfBest = 1e9
    byobbo.setObj(objFit)
    t0 = time.time()
    while True:
        byobbo.optimize(100)

        byobboo = byobbo.getBBO()

        # Write the BBO to a json file
        # You could reload and use it with
        #  bbo = BBO.fromDict(json_tricks.loads(open(jsonFn).read()))
        #
        with open(jsonFn, 'w') as f:
            f.write(json_tricks.dumps(bbo.toDict()))

        nIter += 1
        
    byobbo.stop()
    


