import numpy as np

def computeCert(argTuple):
    betaVals, ximesh = argTuple
    d2vals = []
    for i in range(len(ximesh)):
        beta = betaVals[i]
        xi = ximesh[i]
        v = 5
        lr = 2
        rBar = 2
        sigma = 0.4
        deltaFMax = np.pi / 4
        betaMax = np.arctan(0.5 * np.tan(deltaFMax))

        h = (sigma * np.cos(xi/2) + 1 - sigma)/rBar - 1 / (rBar + 1e-6)
        Lh = v * ((1 / rBar**2) * np.cos(xi - beta)
                  + sigma * np.sin(xi/2) * np.sin(xi - beta) / (2 * rBar**2)
                  + sigma * np.sin(xi/2) * np.sin(beta) / (2 * rBar * lr))

        delta_max = (betaMax - beta) / (Lh + 1e-6) if Lh > 0 else 0
        d2vals.append(delta_max)
    return d2vals
