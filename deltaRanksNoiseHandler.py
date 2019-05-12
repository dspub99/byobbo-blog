

import numpy as np
import scipy.stats as ss


class DeltaRanksNoiseHandler:
    def __init__(self, seed, nEvalsPerW, delta0=.25, sigmaBoost=1.1, nBoot=10):
        if nEvalsPerW==1:
            print ("DeltaRanksNoiseHandler is off because nEvalsPerW=1")
        else:
            assert(nEvalsPerW>=3), "Can't run bootstrap with too few samples"
        self._rng = np.random.RandomState(seed)
        self._delta0 = delta0
        self._nBoot = nBoot
        self._nEvalsPerW = int(nEvalsPerW)
        self._sigmaBoost = sigmaBoost
        
        self._cSigma = 1
        self._pIncEvals = .1
        
    def showConfig(self):
        if self._delta0 is None:
            sdelta0 = "n/a"
        else:
            sdelta0 = "%.4f" % self._delta0
        print ("%s: nEvalsPerW = %d delta0 = %s nBoot = %d sigmaBoost = %.3f" % (self.__class__.__name__, self._nEvalsPerW, sdelta0, self._nBoot, self._sigmaBoost))

    def getSigmaCoef(self):
        return self._cSigma
        
    def getNEvalsPerW(self):
        return self._nEvalsPerW

    def _bootRank(self, evals):
        i = self._rng.randint(evals.shape[1], size=(evals.shape[1],))
        phis = evals[:,i].mean(axis=1)
        return ss.rankdata(phis, method='dense')

    def tell(self, phis):
        if self._nEvalsPerW==1:
            return
        ranks = []
        for _ in range(self._nBoot):
            ranks.append(self._bootRank(phis))
        ranks = np.array(ranks).T
        delta = ranks.std(axis=1)
        mrc = delta.max()
        self._mrc = mrc
        if self._delta0 is None:
            raise Exception("delta0 is None!")
            self._theta = phis.shape[0]
            return True

        
        self._theta = self._delta0  * phis.shape[0]
        self._cSigma = 1

        if mrc > self._theta:
            self._cSigma = self._sigmaBoost

     
    def getKRC(self):
        return self._mrc /  self._theta
            
    def __str__(self):
        return "mrc = %.2f kRC = %.3f cSig = %.3f" % (self._mrc, self.getKRC(), self._cSigma)

def test_drnh():
    print ()
    d = DeltaRanksNoiseHandler(17, nEvalsPerW=10, delta0=1, sigmaBoost=1.1, nBoot=10)
    evals = np.random.normal(size=(3,5))
    d.tell(evals)
    print (str(d))
    
    
