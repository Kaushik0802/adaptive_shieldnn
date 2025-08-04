import os
import pickle
import numpy as np
import scipy.optimize
import deriv_definitions as dd
import multiprocessing
from multiprocessing import Manager, Lock
import time

if __name__ == '__main__':

    with open('deriv3_certs.p','rb') as fp:
        deriv3_certs = pickle.load(fp)
    with open('deriv2_certs.p','rb') as fp:
        deriv2_certs = pickle.load(fp)
    with open('deriv1_certs.p','rb') as fp:
        deriv1_certs = pickle.load(fp)
    
    lr = deriv3_certs[0]['lr']
    sigma = deriv3_certs[0]['sigma']
    rBar = deriv3_certs[0]['rBar']
    vmax=1
    h  = lambda v,r,xi     : (sigma*np.cos(xi/2) + 1 - sigma)/rBar - 1/r
    Lh = lambda v,r,xi,beta: v*( \
            1/(r**2)*np.cos(xi-beta) + sigma*np.sin(xi/2)*np.sin(xi-beta)/(2*rBar*r) + sigma*np.sin(xi/2)*np.sin(beta)/(2*rBar*lr) \
        )
    barrier = lambda xi: rBar/(sigma*np.cos(xi/2) + 1 - sigma)
    alpha = lambda x: 2.5*vmax*sigma*x/(2*rBar*lr)
    LhHat = lambda xi,beta : Lh(1,barrier(xi),xi,beta)

    K3 = dd.deriv3NumBoundLambda(sigma,lr,rBar) * np.max([ 1/((1-d['margin'])*d['guaranteedMin']) for d in deriv3_certs ])
    K2 = dd.deriv2NumBoundLambda(sigma,lr,rBar) * np.max([ 1/((1-d['margin'])*d['guaranteedMin']) for d in deriv2_certs ])
    K1 = dd.deriv1NumBoundLambda(sigma,lr,rBar) * np.max([ 1/((1-d['margin'])*d['guaranteedMin']) for d in deriv1_certs ])

    lowerExtent = deriv3_certs[0]['lowerExtent']

    refineFactor = 10
    # ximesh = np.linspace(lowerExtent,np.pi,int(np.ceil(refineFactor*K3))+1)
    xiDelta = (np.pi-lowerExtent)/(np.ceil(K3)*refineFactor)
    ximeshLen = int(np.ceil(K3)*refineFactor+1)
    manager = Manager()
    results = manager.Queue()
    progress = manager.Queue()
    progressLock = Lock()
    subsampleFactor = 5000

    def computeCert(idx):
        start_time = time.time()
        xim = np.array([lowerExtent + (idx + k)*xiDelta for k in range(int(np.min([subsampleFactor,ximeshLen-idx])))])
        bVals = np.array([scipy.optimize.newton_krylov(lambda beta: LhHat(x,beta),0, f_tol=1e-10) for x in xim])
        LhValsTemp = LhHat(xim, bVals)
        D2ValsTemp = dd.deriv2( xim, bVals, sigma, lr, rBar)
        start_time = time.time()-start_time
        results.put([ \
            idx, \
            xim[0], \
            bVals[0], \
            np.max(D2ValsTemp + K1*K2*LhValsTemp + 1/refineFactor) \
        ])
        if idx + subsampleFactor >= ximeshLen:
            results.put([ \
                idx + subsampleFactor, \
                xim[-1], \
                bVals[-1], \
                np.max(D2ValsTemp + K1*K2*LhValsTemp + 1/refineFactor) \
            ])
        progress.put(1)
        progressLock.acquire(block=True)
        itsComplete = 0
        while not progress.empty():
            itsComplete = itsComplete + progress.get()
        with open('Progress.txt','w') as fp:
            fp.write(str(int(itsComplete/8848)) + ' % complete' + "\n")
        progress.put(itsComplete)
        progressLock.release()
        return start_time
    
    betaVals = np.ones(int(np.floor(ximeshLen/subsampleFactor))+2 if np.mod(ximeshLen,subsampleFactor)!=1 else int(ximeshLen/subsampleFactor)+1)
    ximeshOut = np.ones(len(betaVals))
    print('Total iterations needed: ' + str(len(betaVals)))
    t = time.time()
    pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count()))
    out = np.array(pool.map(computeCert, range(0,ximeshLen,subsampleFactor)))
    pool.close()
    pool.join()
    t = time.time() - t

    print('Elapsed time: ' + str(t) + ' seconds...')

    certifiedMax = -np.inf
    while not results.empty():
        temp = results.get()
        certifiedMax = np.max([certifiedMax,temp[3]])
        try:
            ximeshOut[int(temp[0]/subsampleFactor)] = temp[1]
            betaVals[int(temp[0]/subsampleFactor)] = temp[2]
        except IndexError:
            print('Index error: tried to access element ' + str(int(temp[0]/subsampleFactor)) + ' in an array of size ' + str(len(ximeshOut)))

    LhVals = LhHat(ximeshOut, betaVals)
    D2Vals = dd.deriv2( ximeshOut, betaVals, sigma, lr, rBar)
    

    retDict = { \
            'refineFactor':refineFactor, \
            'lr':lr, \
            'sigma':sigma, \
            'rBar':rBar, \
            'subsampleFactor': subsampleFactor, \
            'ximesh':ximeshOut, \
            'betaVals':betaVals, \
            'LhVals': LhVals, \
            'D2Vals': D2Vals,\
            'K1':K1, \
            'K2':K2, \
            'K3':K3, \
            'lowerExtent': lowerExtent, \
            'certified_maxValue': certifiedMax, \
            'computationTime': t
        }
    
    with open('D2CertVals.p','wb') as fp:
        pickle.dump(retDict,fp,protocol=pickle.HIGHEST_PROTOCOL)