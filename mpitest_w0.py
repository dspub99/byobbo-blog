#!/usr/bin/env python


from objectives import Sphere
from byobbo import BYOBBO
from bbo import RNNTypes

# BBO._w0 is tuned by CMA-ES
s = Sphere(134, 1)
# nObjIter=1 to get the best answer it can for w0 -- which is x0
bb = BYOBBO(134, rnnType=RNNTypes.Plain, nh=[3,3], nObjIter=1, nReport=1, sigma0=.1, nEvalsPerW=1)
bb.setObj(s)
bb.optimize(300)
bb.stop()
s.reset(0)
print ("X0:", s._x0, bb.getBBO().getW0())
assert( abs(s._x0 - bb.getBBO().getW0()) < .01)




