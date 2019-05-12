#!/usr/bin/env python

import numpy as np
from bbo import BBO, BBOConfig
from perf import MultiPerf

class BYOBBOEval:
    def __init__(self, rnnType, nh, nObjIter, tMax):
        self._rnnType = rnnType
        self._tMax = tMax
        self._nh = nh
        self._nObjIter = nObjIter
        self._offset = 0
        self._perf = MultiPerf()
        self._lastIw = None

    def setOffset(self, offset):
        self._offset = offset
        self._lastIw = None
        
    def setObj(self, obj):
        self._obj = obj
        self._bbo = BBO(BBOConfig(nth=self._obj.nDim, nq=1, nh=self._nh, rnnType=self._rnnType,tMax=self._tMax))
        self._lastIw = None

    def perf(self):
        m = self._perf.means()
        m.extend(self._bbo.perf())
        return m
        
    def setWs(self, ws):
        self._ws = ws
        self._lastIw = None
        
    def wEval(self, iw, ie):
        self._perf.start(0)
        if iw != self._lastIw:
            # setParams is slow, so only call it when needed
            self._bbo.setParams(self._ws[iw])
            self._lastIw = iw
        self._perf.start(1)
        
        self._obj.reset(self._offset + ie)
        xs = []
        ys = []
        for _ in range(self._nObjIter):
            x = self._bbo.query(xs, ys)
            y = self._obj.f(x)
            xs.append(x)
            ys.append([y])
            
        self._perf.start(2)

        ys = (np.array(ys) - self._obj.y0)/self._obj.scale

        # dist from known optimum
        # ret = ((xs[-1] - self._obj._x0)**2).sum()
        
        # total (expensive global opt)
        # ret = ys.mean()

        # exp decay
        if False:
            y = np.log(ys)
            # yy = (y-y.mean()).flatten()
            yy = y.flatten()
            x = np.arange(y.shape[0])
            xx = x-x.mean()
            ret = [(xx*yy).sum() / (xx*xx).sum()]

        if False:
            n = ys.shape[0] - np.arange(ys.shape[0])
            w = np.exp(-n/2)
            ret = [(w*ys).sum() / w.sum()]

        # final
        if True:
            ret = ys[-1]


        # weighted toward end
        if False:
            n = len(ys)
            g = np.log(10) / n
            w = (g ** np.arange(n))[::-1]
            ret = (w * ys.squeeze()).sum() / w.sum()
            
        #ret = np.exp(ys[-1])
        
        # balanced
        # ret = np.log(1e-20 + ys).mean()

        self._perf.stop()
        return ret


 
