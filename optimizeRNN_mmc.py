#!/usr/bin/env python

# You can run this with:
#  mpiexec -n NPROCS optimizeRNN_mmc.py mybbo.json
# where NPROCS is the number of processes you want to spawn,
#   NPROCS <= number of cores

# For example:
#  mpiexec -n 4 optimizeRNN_sphere.py mybbo.json
#


if __name__=="__main__":

    import time
    import sys

    import numpy as np
    import json_tricks

    from byobbo import BYOBBO
    from bbo import RNNTypes
    from mmcObjective import MMCObjective

    jsonFn = sys.argv[1]

    nObjIter = 3
    seed0 = None
    nEvalsPerW = 300
    sigma0 = .1
    RNNType = RNNTypes.Plain
    NumHidden = [10]*4

    rng = np.random.RandomState(seed0)
    byobbo = BYOBBO(seed=rng.randint(9999), rnnType=RNNType, nh=NumHidden, nObjIter=nObjIter, nReport=1,
                    sigma0=sigma0, nPop = None, tMax=None, nEvalsPerW=nEvalsPerW)
    

    objFit = MMCObjective(rng.randint(9999))
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
        bbo = byobbo.getBBO()
        with open(jsonFn, 'w') as f:
            f.write(json_tricks.dumps(bbo.toDict()))

        nIter += 1
        
    byobbo.stop()
    


