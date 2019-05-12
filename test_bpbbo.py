
def test_bpbbo():
    import numpy as np
    import bpbbo

    numThetas = 3
    numQualities = 1
    nHidden = [4,3]
    np.random.seed(3)
    b = bpbbo.BPBBOPlain(numThetas, numQualities, nHidden)
    p = np.random.normal(size = (b.numParams(),))

    print ("NP:", b.numParams())
    b.setParams(p)
    print ("SET", p)

    b.resize(10)
    n = b.size()
    outThetas = b.getOutThetas()
    for i in range(0, n):
        # get the params
        b.query(i)
        
        # run experiment at these params
        np.copyto(b.getThetas(i), outThetas)
        # record result
        b.getQualities(i)[0] = 2*(i+1)
        
        print ("i = %d" % i, b.getThetas(i), outThetas)

    assert( abs(outThetas[1] == 0.99347009) < 1e-4 )


def test_opt():
    import bpbbo

    runOpt(bpbbo.BPBBOJANET)
    # runOpt(bpbbo.BPBBOPlain)

def runOpt(BPBBO):
    import numpy as np
    import cma


    print ("testOps: %s" % (BPBBO.__name__))
    
    numThetas = 1
    numQualities = 1
    if BPBBO.__name__=="BPBBOPlain":
        nHidden = [3,2]
    else:
        nHidden = [2,1]
    nSteps = 10
    
    def f(x):
        return ( (x-.1)**2 ).sum()

    def fitness(b, p):
        b.setParams(p)

        nSteps = b.size()
        outThetas = b.getOutThetas()
        for i in range(0, nSteps):
            b.query(i)
            np.copyto(b.getThetas(i), outThetas)
            q = f(outThetas[0])
            b.getQualities(i)[0] = q
        return q

    
    b = BPBBO(numThetas, numQualities, nHidden)
    b.resize(nSteps)
    es = cma.CMAEvolutionStrategy([0]*b.numParams(), sigma0=.1, inopts={'seed':1})

    for _ in range(1000):
        ps = es.ask()
        evals = np.array([fitness(b,p) for p in ps])
        es.tell(ps, evals)

    assert( evals.mean()  < 1e-4 )
    
    
    
