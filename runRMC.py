#!/usr/bin/env python

# After optmizing and RNN with something like
#  mpiexec -n 4 optimizeRNN_mmc.py bbo.json
#
# You can test the resultant custom optimizer "out-of-task"
# on RealMountainCar with:
#  ./runRMC.py bbo.json
#
# You can try the optimized BBO from the blog post with
#  ./runRMC.py mmcBBO-blog.json
#

import sys
import json_tricks
from rmc import RealMountainCar
from mmcObjective import Controller
from bbo import BBO

bboJsonFN = sys.argv[1]

bbo = BBO.fromDict(json_tricks.loads(open(bboJsonFN).read()))

ws = []
phis = []
for nEpisode in range(3):
    rmc = RealMountainCar()
    w = bbo.query(ws, phis)
    c = Controller(w)
    phi = 0
    for nFrame in range(1000):
        if rmc.stopped():
            break
        phi += rmc.step(c.force(rmc.state()))
    print ("EVAL: phi = %.3f w = %s" % (phi, w))
    ws.append(w)
    phis.append([-phi])
    
