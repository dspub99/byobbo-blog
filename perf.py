
import time

# times are in seconds

class Perf:
    def __init__(self):
        self._sumDt = 0
        self._n = 0
        self._t0 = None

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        dt = time.perf_counter() - self._t0
        self._sumDt += dt
        self._n += 1
        self._t0 = None
        return dt
    
    def total(self):
        return self._sumDt

    def count(self):
        return self._n
    
    def mean(self):
        if self._n>0:
            return self._sumDt / self._n
        return None


class MultiPerf:
    def __init__(self):
        self._perfs = []
        self._iRunning = None

    def start(self, i):
        self.stop()
        while i >= len(self._perfs):
            self._perfs.append(Perf())
        self._perfs[i].start()
        self._iRunning = i
        
    def stop(self):
        if self._iRunning is not None:
            self._perfs[self._iRunning].stop()
            self._iRunning = None

    def means(self):
        return [p.mean() for p in self._perfs]

    def __len__(self):
        return len(self._perfs)

def test_multiPerf():
    mp = MultiPerf()
    nPerf = 3
    for __ in range(10):
        for i in range(nPerf):
            mp.start(i)
            for _ in range(100):
                time.sleep(.0001*(1+i))
        mp.stop()

    mm = mp.means()
    om = 0
    for i in range(len(mp)):
        m = mm[i]
        print (i, mm[i])
        assert(m > om)
        


def test_perf():
    p = Perf()
    for __ in range(10):
        p.start()
        for _ in range(100):
            time.sleep(.001)
        print (p.mean(), p.stop())
    
    
