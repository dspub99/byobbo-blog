#!/usr/bin/env python

import time

import numpy as np

from leancma import LeanCMA

from mpiEvaluator import MPIEvaluator
from bbo import BBO, BBOConfig
from byobboEval import BYOBBOEval
from perf import Perf, MultiPerf

class BYOBBO:
    def __init__(self, seed, rnnType, nh, nObjIter, nReport, sigma0, nEvalsPerW, nPop=None, tMax=None):
        # switch to worker mode asap
        self._wEval = BYOBBOEval(rnnType, nh, nObjIter, tMax)
        self._tMax = tMax
        self._mpie = MPIEvaluator(self._wEval)
        self._sigma0 = sigma0
        self._nEvalsPerW = nEvalsPerW
        self._nPop = nPop
        self._rnnType = rnnType
        self._nh = nh
        self._rng = np.random.RandomState(seed)
        self._seed = seed+1

        # head node from here on
        print ("BYOBBO: seed = %s nh = %s nObjIter = %d sigma0 = %.4f tMax = %s nPop = %s"
               % (seed, nh, nObjIter, self._sigma0, tMax, nPop))
        self._bbo = None
        self._nIter = 0
        self._trace = []
        self._iReport = 0
        self._nReport = nReport
        self._t0 = None
        self._t = None

        
    def setOffset(self, offset):
        self._mpie.setOffset(offset)
        
    def setObj(self, obj):
        self._obj = obj
        if self._bbo is None:
            self._nDim = obj.nDim
            self._bbo = BBO(BBOConfig(nth=obj.nDim, nq=1, nh=self._nh, rnnType=self._rnnType,tMax=self._tMax))
            self._es = LeanCMA([0]*self._bbo.numParams(), sigma0=self._sigma0, seed=self._seed, nEvalsPerW = self._nEvalsPerW) # TODO: npop
            print ("BYOBBO: nParams = %d" % (self._bbo.numParams()))
            
        self._mpie.setObj(obj)

    def _extractPhi(self, evals):
        # evals has len(ws) rows and nEvalsPerW cols
        newEvals = []
        for row in evals:
            newRow = []
            for phi in row:
                newRow.append(float(phi))
            newEvals.append(newRow)
        return np.array(newEvals)

    def _phi(self, pPhi, ws):

        pPhi.start(0)
        evals = self._mpie.evaluate(ws, self._es.getNEvalsPerW())
        pPhi.start(1)
        evals = self._extractPhi(evals)
        pPhi.start(2)

        self._nEvalsThisTime = evals.shape[1]
        pPhi.stop()
        
        return evals

    def _collectPerf(self):
        [ph, pws] = self._mpie.perf()
        nPerf = len(pws[0])
        
        perf = []
        for i in range(nPerf):
            x = 0
            n = 0
            for pw in pws:
                if len(pw)>i:
                    x += pw[i]
                    n+=1
            x /= n
            perf.append("%.3f" % (1000*x))

        perf = ' '.join(perf)
        ph = ' '.join(["%.3f"%(1000*x) for x in ph])
        return "ph = %s pw = %s" % (ph, perf)

    def optimize(self, nIter):
        if self._t is None:
            self._t0 = time.time()
            self._t = self._t0

        pAsk = Perf()
        pTell = Perf()
        pPhi = MultiPerf()
        for _ in range(nIter):
            pAsk.start()
            ws = np.array(self._es.ask())
            pAsk.stop()
            phis = self._phi(pPhi, ws)
            pTell.start()
            self._es.tell(phis)
            pTell.stop()

            self._iReport += 1
            if self._iReport==self._nReport:
                t = time.time()
                print ("EVAL: iter = %d t = %.2f dt = %.4f phi = %.4e wmn = %.4f wmx = %.4f sigma = %.4e nEvalsPerW = %d" %
                       (self._nIter, t-self._t0, t-self._t, phis.mean(), ws.mean(), np.abs(ws).max(),
                        self._es.getSigma(), self._nEvalsThisTime))
                self._phiLatest = phis.mean()
                
                perf = self._collectPerf()
                print ("PERF: pA = %.3f pT = %.3f pP = %s %s" % (1000*pAsk.mean(), 1000*pTell.mean(), ' '.join(["%.3f"%(1000*p) for p in pPhi.means()]), perf))
                self._t = t
                self._nIter += 1
                self._trace.append(phis.mean())
                self._iReport = 0

        self._wfav = np.array(self._es.getXFavorite())
        self._bbo.setParams(self._wfav)

    def getPhi(self):
        return self._phiLatest
        
    def stop(self):
        self._mpie.stop()

    def getW(self):
        return self._wfav
        
    def getBBO(self):
        return self._bbo

    def getTrace(self):
        return self._trace

