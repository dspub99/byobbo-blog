#!/usr/bin/env python

import numpy as np

from perf import MultiPerf

class RNNTypes:
    Plain = "plain"
    Simple = "simple"
    ResNet = "resnet"
    JANET = "janet"

class BBOConfig:
    def __init__(self, nth, nq, nh, rnnType, tMax):
        self.nth = nth
        self.nq = nq
        self.nh = nh
        self.rnnType = rnnType
        self.tMax = tMax

    def toDict(self):
        d = {'nth': self.nth,
             'nq': self.nq,
             'nh': self.nh,
             'rnnType': self.rnnType,
             'tMax': self.tMax}
        return d

    @staticmethod
    def fromDict(d):
        return BBOConfig(d['nth'], d['nq'], d['nh'], d['rnnType'], d['tMax'])
        
# x in [-1,1]
class BBO:
    version = 3
    def __init__(self, bboConfig):
        if bboConfig.rnnType==RNNTypes.Plain:
            from bpbbo import BPBBOPlain as BPBBO
        elif bboConfig.rnnType==RNNTypes.Simple:
            from bpbbo import BPBBOSimple as BPBBO
        elif bboConfig.rnnType==RNNTypes.ResNet:
            from bpbbo import BPBBORes as BPBBO
        elif bboConfig.rnnType==RNNTypes.JANET:
            from bpbbo import BPBBOJANET as BPBBO
        else:
            raise Exception("Unknown RNNType: %s" % bboConfig.rnnType)
        self._bboc = bboConfig

        self._bpbbo = BPBBO(self._bboc.nth, self._bboc.nq, self._bboc.nh)
        if self._bboc.tMax is not None:
            self._bpbbo.setChronoInit(self._bboc.tMax)
        self._w = None
        self._perf = MultiPerf()
        
    @staticmethod
    def fromDict(d):
        if d['version'] != BBO.version:
            raise Exception("Wrong version %d != %d" % (d['version'], BBO.version))
        bboc = BBOConfig.fromDict(d['bboc'])
        bbo = BBO(bboc)
        if d['w'] is not None:
            bbo.setParams(d['w'])
        return bbo
    
    def toDict(self):
        d = {'bboc': self._bboc.toDict(),
             'w': self._w,
             'version': BBO.version}
        return d
        
    def numParams(self):
        return self._bpbbo.numParams()

    def getW0(self):
        return self._bpbbo.getW0()
    
    def setParams(self, w):
        self._perf.start(0)
        self._w = np.array(w)
        self._perf.start(1)
        self._bpbbo.setParams(self._w)
        self._perf.stop()

    def perf(self):
        return self._perf.means()
        
    def query(self, ths, qs):
        assert(len(ths)==len(qs))

        # TODO: new no-copy interface
        self._perf.start(2)
        n = len(ths)
        self._bpbbo.resize(n)
        for i in range(n):
            np.copyto(self._bpbbo.getThetas(i), np.array(ths[i]))
            np.copyto(self._bpbbo.getQualities(i), np.array(qs[i]))
        self._bpbbo.query(n)
        x = np.array(self._bpbbo.getOutThetas())
        self._perf.stop()
        return x

def test_p0():
    np = 12
    b = BBO(BBOConfig(nth=np,nq=3,nh=[4,3],rnnType=RNNTypes.Plain,tMax=0))
    b.setParams([.1]*b.numParams())
    assert(b.query([], []).shape[0] == np)

def test_counting():
    rng = np.random.RandomState(seed=3)
    
    def test(m,n):
        b = BBO(BBOConfig(nth=m,nq=n,nh=[4,7],rnnType=RNNTypes.Plain, tMax=None))
        w = rng.normal(size=(b.numParams(),))
        b.setParams(w)
        ps = [ rng.normal(size=(m,)) for _ in range(5) ]
        qs = [ rng.normal(size=(n,)) for _ in range(5) ]
        b.query(ps, qs)

    for i in range(1,4):
        test(i,i)
        test(i,1)
        test(1,i)


def test_numParams():
    for nP in range(1,2,3):
        for nh in range(1,2,3):
            for nq in range(1,2,3):
                b = BBO(BBOConfig(nth=nP,nq=nq,nh=[nh], rnnType=RNNTypes.Plain,tMax=5))
                b.setParams([.1] * b.numParams())

                b = BBO(BBOConfig(nth=nP,nq=nq,nh=[nh, nh], rnnType=RNNTypes.Plain,tMax=10))
                b.setParams([.1] * b.numParams())

                b = BBO(BBOConfig(nth=nP,nq=nq,nh=[nh, nh+1], rnnType=RNNTypes.Plain,tMax=0))
                b.setParams([.1] * b.numParams())
                

    
def test_ser():
    import json_tricks
    b = BBO(BBOConfig(nth=7,nq=3,nh=[4,3], rnnType=RNNTypes.Plain,tMax=None))
    d = b.toDict()
    print (d)
    b1 = BBO.fromDict(d)
    d1 = b1.toDict()
    print (d1)
    assert(json_tricks.dumps(d)==json_tricks.dumps(d1))
    
    b.setParams([.1] * b1.numParams())
    d = b.toDict()
    print (d)
    b1 = BBO.fromDict(d)
    d1 = b1.toDict()
    print (d1)
    assert(json_tricks.dumps(d)==json_tricks.dumps(d1))
