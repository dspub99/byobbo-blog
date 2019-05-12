
# Code copied from:
#   https://github.com/CMA-ES/pycma/blob/master/cma/purecma.py
#   https://github.com/CMA-ES/pycma/blob/master/cma/sigma_adaptation.py
# and hacked to support only sep-cma https://hal.inria.fr/inria-00287367/document

# Performance improvement:
#
# cma.CMAEvolutionStrategy on Sphere(nDim=10000) w/CMA_diagonal=True (sep-cma)
#  runs 100 iterations in 40s.
# LeanCMA does it in 2.5
#
# CMAEvolutionStrategy on Sphere(nDim=1000) w/CMA_diagonal=True (sep-cma)
#  gets to f(x)<1e-6 in 1883 iterations in 98s
# LeanCMA gets there in 2076 iterations, 6.6s

# sep-cma-es: https://hal.inria.fr/inria-00287367/document
# TPA: https://hal.inria.fr/inria-00276854v3/document, https://hal.inria.fr/hal-00997294v2/document


import numpy as np
from deltaRanksNoiseHandler import DeltaRanksNoiseHandler

class TPA():
    def __init__(self, rng, nDim):
        self._rng = rng
        self._nDim = nDim
        self._damp = 4
        self._c = 0.3  # rank difference is asymetric and therefore the switch from increase to decrease takes too long
        self._z_exponent = 0.5  # sign(z) * abs(z)**z_exponent, 0.5 seems better with larger popsize, 1 was default
        self._s = 0  # the state/summation variable

    def _norm(self, x):
        return np.sqrt( (x**2).sum() )

    def _hat(self, x):
        return x / self._norm(x)

    def mirroredSamples(self, oxmean, xmean, sigmaCovOver2):
        dxHat = self._hat(xmean - oxmean)
        z0 = self._norm(self._rng.randn(self._nDim)) * dxHat
        z1 = -z0
        xs = []
        xs.append(xmean + sigmaCovOver2 * z0)
        xs.append(xmean + sigmaCovOver2 * z1)
        return xs

    def sigmaUpdate(self, phis):
        f_vals = np.asarray(phis)
        z = sum(f_vals < f_vals[1]) - sum(f_vals < f_vals[0])
        z /= len(f_vals) - 1  # z in [-1, 1]                                                                                                                                                                                                                                       
        self._s = (1 - self._c) * self._s + self._c * np.sign(z) * np.abs(z)**self._z_exponent
        return np.exp(self._s / self._damp)


class CMAESParameters():
    def __init__(self, nDim, popsize=None):
        # Strategy parameter setting: Selection
        if popsize is None:
            self.lam = self._defaultPopSize(nDim)
        else:
            self.lam = popsize

        self.mu = int(self.lam / 2)  # number of parents/points/solutions for recombination

        _weights = np.array([np.log(self.mu + 0.5) - np.log(i+1) if i < self.mu else 0
                             for i in range(self.lam)])
        w_sum = _weights[:self.mu].sum()
        self.weights = _weights / w_sum
        self.sumWeights = self.weights.sum()
        
        # variance-effectiveness of sum w_i x_i
        self.mueff = (self.weights[:self.mu].sum())**2 / (self.weights[:self.mu]**2).sum()
        
        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff/nDim) / (nDim+4 + 2 * self.mueff/nDim)  # time constant for cumulation for cov
        self.omcc = 1 - self.cc
        self.cs = (self.mueff + 2) / (nDim + self.mueff + 5)  # time constant for cumulation for sigma control
        self.omcs = 1 - self.cs
        self.c1 = 2 / ((nDim + 1.3)**2 + self.mueff)  # learning rate for rank-one update of cov
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((nDim + 2)**2 + self.mueff)])  # and for rank-mu update

        # nope: self.cmu *= (nDim + 2)/3  # section two in sep-cma-es paper [https://hal.inria.fr/inria-00287367/document]
        self.cmuSumWeights = self.cmu * self.sumWeights

        self.c1aCoef = self.c1 * self.cc * (2-self.cc)
        self.csnCoef = (self.cs * (2 - self.cs) * self.mueff)**0.5
        self.ccnCoef = (self.cc * (2 - self.cc) * self.mueff)**0.5

        
    def _defaultPopSize(self, nDim):
        return 4 + int(3 * np.log(nDim))


class LeanCMA():
    def __init__(self, x0, sigma0, nEvalsPerW, seed=None):
        self._rng = np.random.RandomState(seed)
        self._nDim = len(x0)
        self._opNDim = 1 + self._nDim
        self._params = CMAESParameters(self._nDim, popsize=None)
        self._tpa = TPA(self._rng, self._nDim)
        self._noiseHandler = DeltaRanksNoiseHandler(seed=self._rng.randint(99999), delta0=.25, nEvalsPerW=nEvalsPerW, sigmaBoost=1.1, nBoot=10)
        
        # initializing dynamic state variables
        self._xmean = np.array(x0[:])  # initial point, distribution mean, a copy
        self._sigma = sigma0
        self._sigmaMax = 2*sigma0
        self._pc = np.zeros(shape=(self._nDim,))   # evolution path for C
        self._ps = np.zeros(shape=(self._nDim,))  # and for sigma
        self._cov = np.ones(shape=(self._nDim,))  # covariance matrix, diagonal only
        self._nEval = 0  # countiter should be equal to _nEval / lam

        
    def getNEvalsPerW(self):
        return self._noiseHandler.getNEvalsPerW()
        
    def ask(self):
        sigmaCovOver2 = self._sigma * self._cov / 2.0
        
        if self._nEval > 0:
            xs = self._tpa.mirroredSamples(self._oxmean, self._xmean, sigmaCovOver2)
        else:
            xs = []

        k0 = len(xs)
        rands = self._rng.randn(self._nDim, self._params.lam-k0)
        for k in range(k0, self._params.lam):
            y = sigmaCovOver2 * rands[:,k-k0]
            xs.append(self._xmean + y)

        xs = np.array(xs)

        self._xsSaved = xs
        return xs

    def tell(self, arg1, arg2=None):
        if arg2 is None:
            phisAll = arg1
        else:
            phisAll = arg2

        self._noiseHandler.tell(phisAll)
        self._sigma *= self._noiseHandler.getSigmaCoef()
        phis = phisAll.mean(axis=1)
        
        self._nEval += len(phis)  # evaluations used within tell
        self._oxmean = self._xmean  # not a copy, xmean is assigned anew later
        xs = self._xsSaved
        self._xsSaved = None
        
        ### Sort by fitness
        xs = xs[np.argsort(phis)]
        
        ### recombination, compute new weighted mean value
        self._xmean = np.dot(xs[:self._params.mu].T,
                            self._params.weights[:self._params.mu])

        ### Cumulation: update evolution paths
        y = (self._xmean - self._oxmean)/self._sigma
        z = y/np.sqrt(self._cov)
        
        self._ps = self._params.omcs * self._ps + self._params.csnCoef * z
        self._pc = self._params.omcc * self._pc

        # hsig
        if (self._ps**2).sum() / self._nDim / (1-self._params.omcs**(2*self._nEval/self._params.lam))   <   2 + 4./self._opNDim:
            self._pc += self._params.ccnCoef * y
            c1a = 1 - self._params.c1aCoef
        else:
            c1a = 1

        ### Adapt covariance matrix cov
        self._cov *= c1a - self._params.cmuSumWeights
        self._cov += self._params.c1 * (self._pc**2)
        
        dx2 = (xs - self._oxmean)**2
        coef = self._params.cmu * self._sigma**2
        self._cov += coef * (dx2 * self._params.weights[:,None]).sum(axis=0).T
        
        self._sigma *= self._tpa.sigmaUpdate(phis)
        self._sigma = min(self._sigmaMax, self._sigma)
        

    def getXFavorite(self):
        return self._xmean
    
    def getSigma(self):
        return self._sigma

    def setSigma(self, sigma):
        self._sigma = sigma

    def getLambda(self):
        return self._params.lam

