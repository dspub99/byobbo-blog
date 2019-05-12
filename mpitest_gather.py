#!/usr/bin/env python


if __name__=="__main__":
    import numpy as np
    from mpiEvaluator import MPIEvaluator
    
    class E:
        def setObj(self, obj):
            pass

        def setWs(self, ws):
            self._ws = ws
            
        def wEval(self, iw, ie):
            return [iw]
    
    me = MPIEvaluator(E())
    if me.isMaster():
        ws = np.random.uniform(-1, 1, size=(17, 3))
        nEvalsPerW = 7
        phis = me.evaluate(ws, nEvalsPerW)
        # print ("PHIS:", phis)
        assert(len(phis) == len(ws))
        for i,row in enumerate(phis):
            assert( len(row) == nEvalsPerW )
            assert( row[0][0] == i )

        me.stop()
