import importlib
import time
import sys
import numpy as np
import pickle
import scipy.optimize
import scipy.special
import scipy


import deriv_definitions as dd
import certifymin




if __name__ == '__main__':

    lr = 2
    deltaFMax = np.pi/4
    betaMax = np.arctan( 0.5*np.tan(deltaFMax) )
    vmax = 20
    rBar = 3.5
    sigma = 0.48

    # Compute the lower extent of xi 
    h  = lambda v,r,xi     : (sigma*np.cos(xi/2) + 1 - sigma)/rBar - 1/r
    Lh = lambda v,r,xi,beta: v*( \
            1/(r**2)*np.cos(xi-beta) + sigma*np.sin(xi/2)*np.sin(xi-beta)/(2*rBar*r) + sigma*np.sin(xi/2)*np.sin(beta)/(2*rBar*lr) \
        )
    barrier = lambda xi: rBar/(sigma*np.cos(xi/2) + 1 - sigma)
    alpha = lambda x: 3*vmax*sigma*x/(2*rBar*lr)

    # Check that there is an admissible control on the barrier
    res = scipy.optimize.root(lambda beta: Lh(vmax,barrier(np.pi),np.pi,beta)+alpha(h(vmax,barrier(np.pi),np.pi)),betaMax)

    if not res.success:
        exit()

    res = scipy.optimize.root(lambda xi: Lh(vmax,barrier(xi),xi,-betaMax)+alpha(h(vmax,barrier(xi),xi)),0)
    if not res.success:
        exit()
    lowerExtent = res.x[0,Ellipsis].tolist()
    if lowerExtent < -np.pi or lowerExtent > np.pi:
        exit()

    lowerExtent = lowerExtent - 1e-3

    deriv3Results = []

    for i in range(len(dd.deriv3DenomLambdasCfuns)):
        dbnd = dd.deriv3DenomBoundsLambdas[i](sigma, lr, rBar)

        run0 = certifymin.CertifyMinimum(lr=lr,rBar=rBar,sigma=sigma,betaMax=betaMax,cfun=dd.deriv3DenomLambdasCfuns[i],dbnd=dbnd,refineFactor=(100 if i==1 or i==2 else 45))
        run0.verifyByTwoLevelAdaptive(lowerExtent, np.pi)

        deriv3Results.append(run0.exportDict())
        deriv3Results[-1]['lowerExtent'] = lowerExtent
        deriv3Results[-1]['upperExtent'] = np.pi

        with open('deriv3_certs.p','wb') as fp:
            pickle.dump(deriv3Results,fp,protocol=pickle.HIGHEST_PROTOCOL)
        
        print('Completed one function item....')
    

    deriv2Results = []

    for i in range(len(dd.deriv2DenomLambdasCfuns)):
        dbnd = dd.deriv2DenomBoundsLambdas[i](sigma, lr, rBar)

        run0 = certifymin.CertifyMinimum(lr=lr,rBar=rBar,sigma=sigma,betaMax=betaMax,cfun=dd.deriv2DenomLambdasCfuns[i],dbnd=dbnd,initGrid=0.00001,refineFactor=50)
        run0.verifyByTwoLevelAdaptive(lowerExtent,np.pi)

        deriv2Results.append(run0.exportDict())
        deriv2Results[-1]['lowerExtent'] = lowerExtent
        deriv2Results[-1]['upperExtent'] = np.pi

        with open('deriv2_certs.p','wb') as fp:
            pickle.dump(deriv2Results,fp,protocol=pickle.HIGHEST_PROTOCOL)
        
        print('Completed one function item....')
    

    deriv1Results = []

    for i in range(len(dd.deriv1DenomLambdasCfuns)):
        dbnd = dd.deriv1DenomBoundsLambdas[i](sigma, lr, rBar)

        run0 = certifymin.CertifyMinimum(lr=lr,rBar=rBar,sigma=sigma,betaMax=betaMax,cfun=dd.deriv1DenomLambdasCfuns[i],dbnd=dbnd,initGrid=0.00001,refineFactor=50)
        run0.verifyByTwoLevelAdaptive(lowerExtent,np.pi)

        deriv1Results.append(run0.exportDict())
        deriv1Results[-1]['lowerExtent'] = lowerExtent
        deriv1Results[-1]['upperExtent'] = np.pi

        with open('deriv1_certs.p','wb') as fp:
            pickle.dump(deriv1Results,fp,protocol=pickle.HIGHEST_PROTOCOL)
        
        print('Completed one function item....')