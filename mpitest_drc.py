#!/usr/bin/env python


from objectives import Sphere
from byobbo import BYOBBO
from bbo import RNNTypes
from deltaRanksNoiseHandler import DeltaRanksNoiseHandler

# BBO._w0 is tuned by CMA-ES
s = Sphere(134, 1)
# nObjIter=1 to get the best answer it can for w0 -- which is x0
nshn = DeltaRanksNoiseHandler(seed=17, nEvalsPerW=10, delta0=.25, nBoot=10)
bb = BYOBBO(134, rnnType=RNNTypes.Plain, nh=[3,3], nObjIter=1, nReport=1, sigma0=.1, nEvalsPerW=10)
bb.setObj(s)
bb.optimize(100)
bb.stop()
s.reset(0)
# assert( nshn.getKRC() < 1)


