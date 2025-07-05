import os
import pickle
import numpy as np
import scipy.optimize
import deriv_definitions as dd
import multiprocessing
from tqdm import tqdm
import time


def worker(args):
    idx, lowerExtent, xiDelta, ximeshLen, subsampleFactor, sigma, lr, rBar, K1, K2, refineFactor = args

    def barrier(xi):
        return rBar / (sigma * np.cos(xi / 2) + 1 - sigma)

    def Lh(v, r, xi, beta):
        return v * (
            1 / (r ** 2) * np.cos(xi - beta)
            + sigma * np.sin(xi / 2) * np.sin(xi - beta) / (2 * rBar * r)
            + sigma * np.sin(xi / 2) * np.sin(beta) / (2 * rBar * lr)
        )

    def LhHat(xi, beta):
        return Lh(1, barrier(xi), xi, beta)

    xim = np.array([
        lowerExtent + (idx + k) * xiDelta
        for k in range(int(min(subsampleFactor, ximeshLen - idx)))
    ])

    bVals = np.array([
        scipy.optimize.newton_krylov(lambda beta: LhHat(x, beta), 0, f_tol=1e-10)
        for x in xim
    ])
    LhValsTemp = np.array([LhHat(x, b) for x, b in zip(xim, bVals)])
    D2ValsTemp = dd.deriv2(xim, bVals, sigma, lr, rBar)

    return [
        idx,
        xim[0],
        bVals[0],
        np.max(D2ValsTemp + K1 * K2 * LhValsTemp + 1 / refineFactor)
    ]


if __name__ == '__main__':
    # Load cert data
    with open('deriv3_certs.p', 'rb') as fp:
        deriv3_certs = pickle.load(fp)
    with open('deriv2_certs.p', 'rb') as fp:
        deriv2_certs = pickle.load(fp)
    with open('deriv1_certs.p', 'rb') as fp:
        deriv1_certs = pickle.load(fp)

    lr = deriv3_certs[0]['lr']
    sigma = deriv3_certs[0]['sigma']
    rBar = deriv3_certs[0]['rBar']

    K3 = dd.deriv3NumBoundLambda(sigma, lr, rBar) * max(
        1 / ((1 - d['margin']) * d['guaranteedMin']) for d in deriv3_certs
    )
    K2 = dd.deriv2NumBoundLambda(sigma, lr, rBar) * max(
        1 / ((1 - d['margin']) * d['guaranteedMin']) for d in deriv2_certs
    )
    K1 = dd.deriv1NumBoundLambda(sigma, lr, rBar) * max(
        1 / ((1 - d['margin']) * d['guaranteedMin']) for d in deriv1_certs
    )

    lowerExtent = deriv3_certs[0]['lowerExtent']
    refineFactor = 2 #initially 10
    xiDelta = (np.pi - lowerExtent) / (np.ceil(K3) * refineFactor)
    ximeshLen = int(np.ceil(K3) * refineFactor + 1)
    subsampleFactor = 50 # intially 5000

    betaVals = np.ones(
        int(np.floor(ximeshLen / subsampleFactor)) + 2
        if ximeshLen % subsampleFactor != 1
        else int(ximeshLen / subsampleFactor) + 1
    )
    ximeshOut = np.ones(len(betaVals))

    print("Total iterations needed:", len(betaVals))

    # Prepare arguments
    args_list = [
        (i, lowerExtent, xiDelta, ximeshLen, subsampleFactor, sigma, lr, rBar, K1, K2, refineFactor)
        for i in range(0, ximeshLen, subsampleFactor)
    ]

    t0 = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(worker, args_list), total=len(args_list)))

    t_elapsed = time.time() - t0
    print(f"Elapsed time: {t_elapsed:.2f} seconds")

    # Process results
    certifiedMax = -np.inf
    for temp in results:
        idx = int(temp[0] / subsampleFactor)
        certifiedMax = max(certifiedMax, temp[3])
        try:
            ximeshOut[idx] = temp[1]
            betaVals[idx] = temp[2]
        except IndexError:
            print(f"Index error at idx {idx}, mesh size = {len(ximeshOut)}")

    # Final values
    def barrier(xi): return rBar / (sigma * np.cos(xi / 2) + 1 - sigma)

    def Lh(v, r, xi, beta):
        return v * (
            1 / (r ** 2) * np.cos(xi - beta)
            + sigma * np.sin(xi / 2) * np.sin(xi - beta) / (2 * rBar * r)
            + sigma * np.sin(xi / 2) * np.sin(beta) / (2 * rBar * lr)
        )

    def LhHat(xi, beta): return Lh(1, barrier(xi), xi, beta)

    LhVals = np.array([LhHat(x, b) for x, b in zip(ximeshOut, betaVals)])
    D2Vals = dd.deriv2(ximeshOut, betaVals, sigma, lr, rBar)

    retDict = {
        'refineFactor': refineFactor,
        'lr': lr,
        'sigma': sigma,
        'rBar': rBar,
        'subsampleFactor': subsampleFactor,
        'ximesh': ximeshOut,
        'betaVals': betaVals,
        'LhVals': LhVals,
        'D2Vals': D2Vals,
        'K1': K1,
        'K2': K2,
        'K3': K3,
        'lowerExtent': lowerExtent,
        'certified_maxValue': certifiedMax,
        'computationTime': t_elapsed
    }

    with open('D2CertVals.p', 'wb') as fp:
        pickle.dump(retDict, fp, protocol=pickle.HIGHEST_PROTOCOL)
