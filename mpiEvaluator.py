#!/usr/bin/env python

import sys
import numpy as np
from mpi4py import MPI

from perf import MultiPerf

# mpirun -n 4 python script.py

CMD_OFF = "__MPIEvaluatorOFFSET__"
CMD_PERF = "__MPIEvaluatorPERF__"
CMD_OBJS = "__MPIEvaluatorOBJS__"
CMD_WEVAL = "__MPIEvaluatorEVAL__"
CMD_WDONE = "__MPIEvaluatorDONE__"

class MPIEvalWorker:
    def __init__(self, evaluator, comm, rank):
        self._comm = comm
        self._rank = rank
        self._evaluator = evaluator

    def run(self):
        self._comm.bcast(None, root=0)
        
        while True:
            cmd = self._comm.scatter(None, root=0)
            
            if cmd==CMD_WDONE:
                return
            if cmd==CMD_OBJS:
                payload = self._comm.scatter(None, root=0)
                self._evaluator.setObj(payload)
            elif cmd==CMD_OFF:
                offset = int(self._comm.scatter(None, root=0))
                self._evaluator.setOffset(offset)
            elif cmd==CMD_PERF:
                self._comm.gather(self._evaluator.perf(), root=0)
            elif cmd==CMD_WEVAL:
                wsShape = self._comm.bcast(None, root=0)
                ws = np.empty(shape=wsShape)
                self._comm.Bcast(ws, root=0)
                self._evaluator.setWs(ws)
                
                job = self._comm.scatter(None, root=0)
                res = []
                for iw, ie in job:
                    res.append( self._evaluator.wEval(iw, ie) )

                self._comm.gather(res, root=0)
            else:
                raise Exception("Unknown command [%s]" % str(cmd))
                    
class MPIEvaluator:
    def __init__(self, evaluator):
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()
        
        if self._rank>0:
            MPIEvalWorker(evaluator, self._comm, self._rank).run()
            sys.exit(0)

        self._nWorkers = self._size-1
        self._init()
        self._perf = MultiPerf()

    def isMaster(self):
        return (self._rank==0)
        
    def _init(self):
        self._comm.bcast("Initialize", root=0)

    def _sendAllCmd(self, cmd):
        cmd = [cmd]*(1+self._nWorkers)
        self._comm.scatter(cmd, root=0)
        
    def _sendAll(self, cmd, payload):
        self._sendAllCmd(cmd)
        payload = [payload]*(1+self._nWorkers)
        self._comm.scatter(payload, root=0)
        
    def setOffset(self, offset):
        self._sendAll(CMD_OFF, offset)
        
    def setObj(self, obj):
        self._sendAll(CMD_OBJS, obj)

    def perf(self):
        self._sendAllCmd(CMD_PERF)
        pw = self._comm.gather(None, root=0)[1:]
        return (self._perf.means(), pw)
    
    def evaluate(self, ws, nEvalsPerW):
        self._perf.start(0)
        cmd = [CMD_WEVAL]*(1 + self._nWorkers)
        self._comm.scatter(cmd, root=0)
        self._comm.bcast(ws.shape, root=0)
        self._comm.Bcast(ws, root=0)

        tasks = []
        for iw in range(len(ws)):
            for ie in range(nEvalsPerW):
                tasks.append( (iw, ie) )
        nTasksPerWorker = int( len(tasks) / self._nWorkers) + 1

        jobs = []
        iTask = 0
        for iwrk in range(self._nWorkers):
            iMax = min([iTask + nTasksPerWorker, len(tasks)])
            jobs.append( tasks[iTask:iMax] )
            iTask += iMax-iTask
        assert(iTask == len(tasks)), "%d != %d" % (iTask, len(tasks))
        assert(len(jobs) == self._nWorkers)
        if True:
            ntasks = 0
            for j in jobs:
                ntasks += len(j)
            assert(ntasks == len(tasks)), "%d != %d" % (ntasks, len(tasks))

        jobs.insert(0, None)
        self._comm.scatter(jobs, root=0)
        
        self._perf.start(1)
        phis = []
        for res in self._comm.gather(None, root=0)[1:]:
            phis.extend(res)


        self._perf.start(2)
        newPhis = []
        i = 0
        for _ in range(len(ws)):
            row = []
            for _ in range(nEvalsPerW):
                row.append(phis[i])
                i += 1
            newPhis.append(row)

        self._perf.stop()
        return newPhis

        

    def stop(self):
        assert(self.isMaster())
        self._comm.scatter([CMD_WDONE]*self._size, root=0)

if __name__ == "__main__":

    class E:
        def setObj(self, obj):
            pass

        def setWs(self, ws):
            print ("SETWS", ws.shape)
            self._ws = ws
            
        def wEval(self, iw):
            #w = self._ws[iw,:]
            #return ( (w-.5)**2 ).mean()
            return iw
    
    me = MPIEvaluator(E())
    if me.isMaster():
        ws = np.random.uniform(-1, 1, size=(17, 3))
        phis = me.evaluate(ws, 7)
        print ("PHIS:", phis)
        i = np.where(phis==phis.min())[0]

        me.stop()
        
    
