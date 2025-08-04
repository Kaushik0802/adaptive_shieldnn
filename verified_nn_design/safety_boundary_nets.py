# from keras.models import Sequential, Model
# from keras.layers import Dense, Activation, Input
# from keras import regularizers
# from keras import initializers
# import tensorflow as tf
# from keras import backend as K
# import keras
import numpy as np
import scipy.optimize
import scipy.special
import scipy
import matplotlib.pyplot as plt
import re
from multiprocessing import Pool
import importlib
import time
import sys

import deriv_definitions as dd
import certifymin

# Symbols to be imported from deriv_definitions.py:
#
# Definitions associated with the second derivative:
#
#	* deriv2: function to calculate the second derivative along constant contours of Lh
#	
# 	The rest of these definitions pretain to derivatives of deriv2 *with respect to beta*. In particular, 
#	we first expand that derivative into terms whose numerators are products of constants and trig
#	functions; the denominators of those terms then determine whether or not this derivative is bounded
#	above.
#
#	* deriv2NumBoundLambda: upper bound on the sum of numerators in the **beta derivative of deriv2**
#	* deriv2DenomLambdas: tuple of lambda functions, each of which computes the denominator of some term
#		from the **beta derivative of deriv2**
#	* deriv2DenomBoundsLambdas: tuple of lambda functions, each of which computes an upper bound for
#		the 2-norm of the derivative (w.r.t. the vector [beta,xi]) of the corresponding element in 
#		deriv2DenomLambdas
#
#
# Definitions associated with the third derivative:
#
#	* deriv3: function to calculate the third derivative along constant contours of Lh
#	* deriv3NumBoundLambda: upper bound on the sum of numerators in **deriv3** (contrast with definition above)
#	* deriv3DenomLambdas: tuple of lambda functions, each of which computes the denominator of some term
#		from the **deriv3** (contrast with definition above)
#	* deriv3DenomBoundsLambdas: tuple of lambda functions, each of which computes an upper bound for
#		the 2-norm of the derivative (w.r.t. the vector [beta,xi]) of the corresponding element in 
#		deriv3DenomLambdas

def verifyParameters(deltaFMax,vmax,rBar,sigma,lr):
	betaMax = np.arctan( 0.5*np.tan(deltaFMax) )
	h  = lambda v,r,xi     : (sigma*np.cos(xi/2) + 1 - sigma)/rBar - 1/r
	Lh = lambda v,r,xi,beta: v*( \
			1/(r**2)*np.cos(xi-beta) + sigma*np.sin(xi/2)*np.sin(xi-beta)/(2*rBar*r) + sigma*np.sin(xi/2)*np.sin(beta)/(2*rBar*lr) \
		)
	barrier = lambda xi: rBar/(sigma*np.cos(xi/2) + 1 - sigma)
	alpha = lambda x: 3*vmax*sigma*x/(2*rBar*lr)

	# Check that there is an admissible control on the barrier
	res = scipy.optimize.root(lambda beta: Lh(vmax,barrier(np.pi),np.pi,beta)+alpha(h(vmax,barrier(np.pi),np.pi)),betaMax)

	if not res.success:
		return False
	
	res = scipy.optimize.root(lambda xi: Lh(vmax,barrier(xi),xi,-betaMax)+alpha(h(vmax,barrier(xi),xi)),0)
	if not res.success:
		return False
	lowerExtent = res.x[0,Ellipsis].tolist()
	if lowerExtent < -np.pi or lowerExtent > np.pi:
		return False

	# breakpoint()

	# Our task here is to show that the deriv2 is negative for all (xi,beta) values along the "zero" contour of Lh.
	# However, we can only use a numerical zero-finding algorithm to **approxiately** get those values along
	# such a curve. Thus, we have two tasks:
	#	1) use information from the "deriv2" expressions to bound the error of deriv2 at the point returned
	#		by the numerical root-finding algorithm (as a function of how close the numerical root is to
	#		the actual root).
	#	2) iteratively refine a xi mesh and run the numerical root-finding algorithm until the mesh is fine
	#		enough that we can be sure balls centered on the numerical root (with radius defined by the largest
	#		value of deriv3) at one xi mesh point contain the actual root at the neighboring xi mesh points

	d3bnd = np.max([ denomBound(sigma, lr, rBar) for denomBound in dd.deriv3DenomBoundsLambdas ])


	breakpoint()

	i = 17
	d3bnd = dd.deriv3DenomBoundsLambdas[i](sigma, lr, rBar)
	denom = dd.deriv3DenomLambdas[i]

	run0 = certifymin.CertifyMinimum(lr=lr,rBar=rBar,sigma=sigma,betaMax=betaMax,cfun=dd.deriv3DenomLambdasCfuns[i],dbnd=d3bnd,refineFactor=45)

	# breakpoint()

	run0.verifyByTwoLevelAdaptive(lowerExtent,np.pi)

	breakpoint()
		



def rootFinderPrecision(xi):
	return 1e-10

if __name__ == '__main__':
	importlib.reload(dd)
	importlib.reload(certifymin)
	# Bicycle/barrier parameters
	lr = 2
	deltaFMax = np.pi/4
	betaMax = np.arctan( 0.5*np.tan(deltaFMax) )
	vmax = 20
	rBar = 3.5
	sigma = 0.48

	h  = lambda v,r,xi     : (sigma*np.cos(xi/2) + 1 - sigma)/rBar - 1/r
	Lh = lambda v,r,xi,beta: v*( \
		1/(r**2)*np.cos(xi-beta) + sigma*np.sin(xi/2)*np.sin(xi-beta)/(2*rBar*r) + sigma*np.sin(xi/2)*np.sin(beta)/(2*rBar*lr) \
		)
	barrier = lambda xi: rBar/(sigma*np.cos(xi/2) + 1 - sigma)
	alpha = lambda x: 2.5*vmax*sigma*x/(2*rBar*lr)

	verifyParameters(deltaFMax,vmax,rBar,sigma,lr)










































