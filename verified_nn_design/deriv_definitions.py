import numpy as np


def deriv2(xi,beta,sigma,lr,rBar):
	return ( \
		sigma*(8*rBar*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(2*beta) + \
		(8*(2*sigma*(-2*rBar**2 + 16*lr*rBar*(-1 + sigma) + \
		lr**2*(46 - 92*sigma + 57*sigma**2))*np.cos(xi/2) + \
		lr*(32*lr - 8*rBar - 96*lr*sigma + 16*rBar*sigma + 142*lr*sigma**2 - \
		13*rBar*sigma**2 - 78*lr*sigma**3 + (-42*lr*(-1 + sigma)*sigma**2 + \
		rBar*(-24 + 48*sigma - 38*sigma**2))*np.cos(xi) + 6*sigma*(4*rBar*(-1 + sigma) + \
		lr*sigma**2)*np.cos((3*xi)/2) - 5*rBar*sigma**2*np.cos(2*xi)))*(2*lr*sigma**2*np.sin(beta - 2*xi) - \
		9*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 8*lr*np.sin(beta - xi) - \
		16*lr*sigma*np.sin(beta - xi) + 12*lr*sigma**2*np.sin(beta - xi) + \
		5*lr*sigma*np.sin(beta - xi/2) + rBar*sigma*np.sin(beta - xi/2) - \
		5*lr*sigma**2*np.sin(beta - xi/2) + rBar*sigma*np.sin(beta + xi/2))*np.sin(xi/2))/(3*lr*sigma**2*np.sin(beta) + \
		lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + \
		2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2)) - \
		lr*(-3*lr*(-1 + sigma)*sigma**2*np.sin(2*beta - (7*xi)/2) + \
		8*lr*sigma*(2 - 4*sigma + 3*sigma**2)*np.sin(2*beta - 3*xi) + \
		24*lr*np.sin(2*beta - (5*xi)/2) - 72*lr*sigma*np.sin(2*beta - (5*xi)/2) + \
		129*lr*sigma**2*np.sin(2*beta - (5*xi)/2) + 9*rBar*sigma**2*np.sin(2*beta - (5*xi)/2) - \
		81*lr*sigma**3*np.sin(2*beta - (5*xi)/2) + 40*lr*np.sin(2*beta - (3*xi)/2) + \
		8*rBar*np.sin(2*beta - (3*xi)/2) - 120*lr*sigma*np.sin(2*beta - (3*xi)/2) - \
		16*rBar*sigma*np.sin(2*beta - (3*xi)/2) + 221*lr*sigma**2*np.sin(2*beta - (3*xi)/2) - 13*rBar*sigma**2*np.sin(2*beta - (3*xi)/2) - \
		141*lr*sigma**3*np.sin(2*beta - (3*xi)/2) + 120*lr*sigma*np.sin(2*(beta - xi)) + \
		24*rBar*sigma*np.sin(2*(beta - xi)) - 240*lr*sigma**2*np.sin(2*(beta - xi)) - \
		24*rBar*sigma**2*np.sin(2*(beta - xi)) + 144*lr*sigma**3*np.sin(2*(beta - xi)) + \
		48*lr*sigma*np.sin(2*beta - xi) - 96*rBar*sigma*np.sin(2*beta - xi) - \
		96*lr*sigma**2*np.sin(2*beta - xi) + 96*rBar*sigma**2*np.sin(2*beta - xi) + \
		72*lr*sigma**3*np.sin(2*beta - xi) - 72*rBar*np.sin(2*beta - xi/2) + \
		144*rBar*sigma*np.sin(2*beta - xi/2) + 15*lr*sigma**2*np.sin(2*beta - xi/2) - \
		105*rBar*sigma**2*np.sin(2*beta - xi/2) - 15*lr*sigma**3*np.sin(2*beta - xi/2) + \
		240*lr*np.sin(xi/2) + 24*rBar*np.sin(xi/2) - 720*lr*sigma*np.sin(xi/2) - \
		48*rBar*sigma*np.sin(xi/2) + 876*lr*sigma**2*np.sin(xi/2) + 42*rBar*sigma**2*np.sin(xi/2) - \
		396*lr*sigma**3*np.sin(xi/2) + 336*lr*sigma*np.sin(xi) + 48*rBar*sigma*np.sin(xi) - \
		672*lr*sigma**2*np.sin(xi) - 48*rBar*sigma**2*np.sin(xi) + 384*lr*sigma**3*np.sin(xi) - \
		24*rBar*np.sin((3*xi)/2) + 48*rBar*sigma*np.sin((3*xi)/2) + 156*lr*sigma**2*np.sin((3*xi)/2) - \
		21*rBar*sigma**2*np.sin((3*xi)/2) - 156*lr*sigma**3*np.sin((3*xi)/2) - 48*rBar*sigma*np.sin(2*xi) + \
		48*rBar*sigma**2*np.sin(2*xi) + 24*lr*sigma**3*np.sin(2*xi) - 15*rBar*sigma**2*np.sin((5*xi)/2) - \
		3*rBar*sigma**2*np.sin((4*beta + xi)/2))))/(64*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
		rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**2 \
	)

def deriv3(xi,beta,sigma,lr,rBar):
	return ( \
		sigma*(-((sigma*(-2*sigma*(-2*rBar**2 + 16*lr*rBar*(-1 + sigma) + \
		lr**2*(46 - 92*sigma + 57*sigma**2))*np.cos(xi/2) + \
		lr*(-32*lr + 8*rBar + 96*lr*sigma - 16*rBar*sigma - 142*lr*sigma**2 + \
		13*rBar*sigma**2 + 78*lr*sigma**3 + (42*lr*(-1 + sigma)*sigma**2 + \
		rBar*(24 - 48*sigma + 38*sigma**2))*np.cos(xi) - 6*sigma*(4*rBar*(-1 + sigma) + \
		lr*sigma**2)*np.cos((3*xi)/2) + \
		5*rBar*sigma**2*np.cos(2*xi)))*np.sin(xi/2)*(8*rBar*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(2*beta) + \
		(8*(2*sigma*(-2*rBar**2 + 16*lr*rBar*(-1 + sigma) + \
		lr**2*(46 - 92*sigma + 57*sigma**2))*np.cos(xi/2) + \
		lr*(32*lr - 8*rBar - 96*lr*sigma + 16*rBar*sigma + 142*lr*sigma**2 - 13*rBar*sigma**2 - \
		78*lr*sigma**3 + (-42*lr*(-1 + sigma)*sigma**2 + rBar*(-24 + 48*sigma - 38*sigma**2))*np.cos(xi) + \
		6*sigma*(4*rBar*(-1 + sigma) + lr*sigma**2)*np.cos((3*xi)/2) - \
		5*rBar*sigma**2*np.cos(2*xi)))*(2*lr*sigma**2*np.sin(beta - 2*xi) - \
		9*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 8*lr*np.sin(beta - xi) - \
		16*lr*sigma*np.sin(beta - xi) + 12*lr*sigma**2*np.sin(beta - xi) + 5*lr*sigma*np.sin(beta - xi/2) + \
		rBar*sigma*np.sin(beta - xi/2) - 5*lr*sigma**2*np.sin(beta - xi/2) + \
		rBar*sigma*np.sin(beta + xi/2))*np.sin(xi/2))/(3*lr*sigma**2*np.sin(beta) + \
		lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - \
		(3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + \
		2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2)) - \
		lr*(-3*lr*(-1 + sigma)*sigma**2*np.sin(2*beta - (7*xi)/2) + 8*lr*sigma*(2 - 4*sigma + \
		3*sigma**2)*np.sin(2*beta - 3*xi) + 24*lr*np.sin(2*beta - (5*xi)/2) - 72*lr*sigma*np.sin(2*beta - (5*xi)/2) + \
		129*lr*sigma**2*np.sin(2*beta - (5*xi)/2) + 9*rBar*sigma**2*np.sin(2*beta - (5*xi)/2) - \
		81*lr*sigma**3*np.sin(2*beta - (5*xi)/2) + 40*lr*np.sin(2*beta - (3*xi)/2) + 8*rBar*np.sin(2*beta - (3*xi)/2) - \
		120*lr*sigma*np.sin(2*beta - (3*xi)/2) - 16*rBar*sigma*np.sin(2*beta - (3*xi)/2) + 221*lr*sigma**2*np.sin(2*beta - (3*xi)/2) - \
		13*rBar*sigma**2*np.sin(2*beta - (3*xi)/2) - 141*lr*sigma**3*np.sin(2*beta - (3*xi)/2) + \
		120*lr*sigma*np.sin(2*(beta - xi)) + 24*rBar*sigma*np.sin(2*(beta - xi)) - 240*lr*sigma**2*np.sin(2*(beta - xi)) - \
		24*rBar*sigma**2*np.sin(2*(beta - xi)) + 144*lr*sigma**3*np.sin(2*(beta - xi)) + 48*lr*sigma*np.sin(2*beta - xi) - \
		96*rBar*sigma*np.sin(2*beta - xi) - 96*lr*sigma**2*np.sin(2*beta - xi) + 96*rBar*sigma**2*np.sin(2*beta - xi) + \
		72*lr*sigma**3*np.sin(2*beta - xi) - 72*rBar*np.sin(2*beta - xi/2) + 144*rBar*sigma*np.sin(2*beta - xi/2) + \
		15*lr*sigma**2*np.sin(2*beta - xi/2) - 105*rBar*sigma**2*np.sin(2*beta - xi/2) - 15*lr*sigma**3*np.sin(2*beta - xi/2) + \
		240*lr*np.sin(xi/2) + 24*rBar*np.sin(xi/2) - 720*lr*sigma*np.sin(xi/2) - 48*rBar*sigma*np.sin(xi/2) + \
		876*lr*sigma**2*np.sin(xi/2) + 42*rBar*sigma**2*np.sin(xi/2) - 396*lr*sigma**3*np.sin(xi/2) + \
		336*lr*sigma*np.sin(xi) + 48*rBar*sigma*np.sin(xi) - 672*lr*sigma**2*np.sin(xi) - 48*rBar*sigma**2*np.sin(xi) + \
		384*lr*sigma**3*np.sin(xi) - 24*rBar*np.sin((3*xi)/2) + 48*rBar*sigma*np.sin((3*xi)/2) + 156*lr*sigma**2*np.sin((3*xi)/2) - \
		21*rBar*sigma**2*np.sin((3*xi)/2) - 156*lr*sigma**3*np.sin((3*xi)/2) - 48*rBar*sigma*np.sin(2*xi) + 48*rBar*sigma**2*np.sin(2*xi) + \
		24*lr*sigma**3*np.sin(2*xi) - 15*rBar*sigma**2*np.sin((5*xi)/2) - 3*rBar*sigma**2*np.sin((4*beta + xi)/2))))/(3*lr*sigma**2*np.sin(beta) + \
		lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + \
		2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))) + \
		(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + \
		lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))*((4*np.cos(xi/2)*(2*sigma*(-2*rBar**2 + \
		16*lr*rBar*(-1 + sigma) + lr**2*(46 - 92*sigma + 57*sigma**2))*np.cos(xi/2) + lr*(32*lr - 8*rBar - 96*lr*sigma + 16*rBar*sigma + \
		142*lr*sigma**2 - 13*rBar*sigma**2 - 78*lr*sigma**3 + (-42*lr*(-1 + sigma)*sigma**2 + \
		rBar*(-24 + 48*sigma - 38*sigma**2))*np.cos(xi) + 6*sigma*(4*rBar*(-1 + sigma) + lr*sigma**2)*np.cos((3*xi)/2) - \
		5*rBar*sigma**2*np.cos(2*xi)))*(2*lr*sigma**2*np.sin(beta - 2*xi) - 9*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 8*lr*np.sin(beta - xi) - \
		16*lr*sigma*np.sin(beta - xi) + 12*lr*sigma**2*np.sin(beta - xi) + 5*lr*sigma*np.sin(beta - xi/2) + \
		rBar*sigma*np.sin(beta - xi/2) - 5*lr*sigma**2*np.sin(beta - xi/2) + rBar*sigma*np.sin(beta + xi/2)))/(3*lr*sigma**2*np.sin(beta) + \
		lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + \
		2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2)) + (16*sigma*(-2*sigma*(-2*rBar**2 + 16*lr*rBar*(-1 + sigma) + \
		lr**2*(46 - 92*sigma + 57*sigma**2))*np.cos(xi/2) + lr*(-32*lr + 8*rBar + 96*lr*sigma - 16*rBar*sigma - 142*lr*sigma**2 + 13*rBar*sigma**2 + \
		78*lr*sigma**3 + (42*lr*(-1 + sigma)*sigma**2 + rBar*(24 - 48*sigma + 38*sigma**2))*np.cos(xi) - 6*sigma*(4*rBar*(-1 + sigma) + lr*sigma**2)*np.cos((3*xi)/2) + \
		5*rBar*sigma**2*np.cos(2*xi)))**2*(2*lr*sigma**2*np.sin(beta - 2*xi) - 9*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 8*lr*np.sin(beta - xi) - \
		16*lr*sigma*np.sin(beta - xi) + 12*lr*sigma**2*np.sin(beta - xi) + 5*lr*sigma*np.sin(beta - xi/2) + rBar*sigma*np.sin(beta - xi/2) - \
		5*lr*sigma**2*np.sin(beta - xi/2) + rBar*sigma*np.sin(beta + xi/2))*np.sin(xi/2)**2)/(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
		6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + \
		2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))**3 + (16*(6*lr*(7*lr*(-1 + sigma)*sigma**2 + \
		rBar*(4 - 8*sigma + 8*sigma**2))*np.cos(xi/2) + sigma*(-23*lr**2 + 26*lr*rBar + rBar**2 + 46*lr**2*sigma - \
		26*lr*rBar*sigma - 33*lr**2*sigma**2 - 9*lr*(4*rBar*(-1 + sigma) + lr*sigma**2)*np.cos(xi) + \
		10*lr*rBar*sigma*np.cos((3*xi)/2)))*(2*lr*sigma**2*np.sin(beta - 2*xi) - 9*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 8*lr*np.sin(beta - xi) - \
		16*lr*sigma*np.sin(beta - xi) + 12*lr*sigma**2*np.sin(beta - xi) + 5*lr*sigma*np.sin(beta - xi/2) + rBar*sigma*np.sin(beta - xi/2) - \
		5*lr*sigma**2*np.sin(beta - xi/2) + rBar*sigma*np.sin(beta + xi/2))*np.sin(xi/2)**2)/(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
		6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - \
		2*rBar*sigma*np.sin(beta + xi/2)) - (4*rBar*(rBar - 5*lr*(-1 + sigma))*sigma*np.cos(2*beta)*(2*lr*sigma**2*np.sin(beta - 2*xi) - \
		9*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 8*lr*np.sin(beta - xi) - 16*lr*sigma*np.sin(beta - xi) + 12*lr*sigma**2*np.sin(beta - xi) + \
		5*lr*sigma*np.sin(beta - xi/2) + rBar*sigma*np.sin(beta - xi/2) - 5*lr*sigma**2*np.sin(beta - xi/2) + \
		rBar*sigma*np.sin(beta + xi/2)))/(-2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) + rBar*sigma*np.cos(beta)*np.sin(xi/2) - \
		lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2)) + (2*sigma*(-2*sigma*(-2*rBar**2 + 16*lr*rBar*(-1 + sigma) + \
		lr**2*(46 - 92*sigma + 57*sigma**2))*np.cos(xi/2) + lr*(-32*lr + 8*rBar + 96*lr*sigma - 16*rBar*sigma - 142*lr*sigma**2 + 13*rBar*sigma**2 + \
		78*lr*sigma**3 + (42*lr*(-1 + sigma)*sigma**2 + rBar*(24 - 48*sigma + 38*sigma**2))*np.cos(xi) - \
		6*sigma*(4*rBar*(-1 + sigma) + lr*sigma**2)*np.cos((3*xi)/2) + \
		5*rBar*sigma**2*np.cos(2*xi)))*np.sin(xi/2)*(-8*rBar*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(2*beta) + \
		lr*(-3*lr*(-1 + sigma)*sigma**2*np.sin(2*beta - (7*xi)/2) + 8*lr*sigma*(2 - 4*sigma + 3*sigma**2)*np.sin(2*beta - 3*xi) + \
		24*lr*np.sin(2*beta - (5*xi)/2) - 72*lr*sigma*np.sin(2*beta - (5*xi)/2) + 129*lr*sigma**2*np.sin(2*beta - (5*xi)/2) + \
		9*rBar*sigma**2*np.sin(2*beta - (5*xi)/2) - 81*lr*sigma**3*np.sin(2*beta - (5*xi)/2) + 40*lr*np.sin(2*beta - (3*xi)/2) + \
		8*rBar*np.sin(2*beta - (3*xi)/2) - 120*lr*sigma*np.sin(2*beta - (3*xi)/2) - 16*rBar*sigma*np.sin(2*beta - (3*xi)/2) + \
		221*lr*sigma**2*np.sin(2*beta - (3*xi)/2) - 13*rBar*sigma**2*np.sin(2*beta - (3*xi)/2) - 141*lr*sigma**3*np.sin(2*beta - (3*xi)/2) + \
		120*lr*sigma*np.sin(2*(beta - xi)) + 24*rBar*sigma*np.sin(2*(beta - xi)) - 240*lr*sigma**2*np.sin(2*(beta - xi)) - \
		24*rBar*sigma**2*np.sin(2*(beta - xi)) + 144*lr*sigma**3*np.sin(2*(beta - xi)) + 48*lr*sigma*np.sin(2*beta - xi) - \
		96*rBar*sigma*np.sin(2*beta - xi) - 96*lr*sigma**2*np.sin(2*beta - xi) + 96*rBar*sigma**2*np.sin(2*beta - xi) + \
		72*lr*sigma**3*np.sin(2*beta - xi) - 72*rBar*np.sin(2*beta - xi/2) + 144*rBar*sigma*np.sin(2*beta - xi/2) + \
		15*lr*sigma**2*np.sin(2*beta - xi/2) - 105*rBar*sigma**2*np.sin(2*beta - xi/2) - 15*lr*sigma**3*np.sin(2*beta - xi/2) + \
		240*lr*np.sin(xi/2) + 24*rBar*np.sin(xi/2) - 720*lr*sigma*np.sin(xi/2) - 48*rBar*sigma*np.sin(xi/2) + 876*lr*sigma**2*np.sin(xi/2) + \
		42*rBar*sigma**2*np.sin(xi/2) - 396*lr*sigma**3*np.sin(xi/2) + 336*lr*sigma*np.sin(xi) + 48*rBar*sigma*np.sin(xi) - 672*lr*sigma**2*np.sin(xi) - \
		48*rBar*sigma**2*np.sin(xi) + 384*lr*sigma**3*np.sin(xi) - 24*rBar*np.sin((3*xi)/2) + 48*rBar*sigma*np.sin((3*xi)/2) + 156*lr*sigma**2*np.sin((3*xi)/2) - \
		21*rBar*sigma**2*np.sin((3*xi)/2) - 156*lr*sigma**3*np.sin((3*xi)/2) - 48*rBar*sigma*np.sin(2*xi) + 48*rBar*sigma**2*np.sin(2*xi) + \
		24*lr*sigma**3*np.sin(2*xi) - 15*rBar*sigma**2*np.sin((5*xi)/2) - 3*rBar*sigma**2*np.sin((4*beta + xi)/2))))/(3*lr*sigma**2*np.sin(beta) + \
		lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + \
		2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))**2 + (lr*(-24*sigma*(3*rBar**2*(2 - 4*sigma + 3*sigma**2) - \
		2*lr*rBar*(-56 + 168*sigma - 205*sigma**2 + 93*sigma**3) + 2*lr**2*(131 - 524*sigma + 882*sigma**2 - 716*sigma**3 + 231*sigma**4))*np.sin(beta) + \
		216*rBar**2*sigma*(2 - 4*sigma + 3*sigma**2)*np.sin(3*beta) - 3*lr**2*sigma**4*np.sin(3*beta - (11*xi)/2) + 3*lr**2*sigma**5*np.sin(3*beta - (11*xi)/2) - \
		14*lr**2*sigma**3*np.sin(3*beta - 5*xi) + 28*lr**2*sigma**4*np.sin(3*beta - 5*xi) - 30*lr**2*sigma**5*np.sin(3*beta - 5*xi) + \
		75*lr*rBar*sigma**4*np.sin(beta - (9*xi)/2) - 135*lr**2*sigma**4*np.sin(3*beta - (9*xi)/2) - 27*lr*rBar*sigma**4*np.sin(3*beta - (9*xi)/2) + \
		135*lr**2*sigma**5*np.sin(3*beta - (9*xi)/2) + 696*lr*rBar*sigma**3*np.sin(beta - 4*xi) - 696*lr*rBar*sigma**4*np.sin(beta - 4*xi) - \
		96*lr**2*sigma**5*np.sin(beta - 4*xi) + 112*lr**2*sigma*np.sin(3*beta - 4*xi) - 448*lr**2*sigma**2*np.sin(3*beta - 4*xi) + \
		256*lr**2*sigma**3*np.sin(3*beta - 4*xi) - 120*lr*rBar*sigma**3*np.sin(3*beta - 4*xi) + 384*lr**2*sigma**4*np.sin(3*beta - 4*xi) + \
		120*lr*rBar*sigma**4*np.sin(3*beta - 4*xi) - 336*lr**2*sigma**5*np.sin(3*beta - 4*xi) + 2080*lr*rBar*sigma**2*np.sin(beta - (7*xi)/2) - \
		4160*lr*rBar*sigma**3*np.sin(beta - (7*xi)/2) - 1107*lr**2*sigma**4*np.sin(beta - (7*xi)/2) + 2499*lr*rBar*sigma**4*np.sin(beta - (7*xi)/2) + \
		1107*lr**2*sigma**5*np.sin(beta - (7*xi)/2) + 192*lr**2*np.sin(3*beta - (7*xi)/2) - 960*lr**2*sigma*np.sin(3*beta - (7*xi)/2) + \
		1472*lr**2*sigma**2*np.sin(3*beta - (7*xi)/2) - 128*lr*rBar*sigma**2*np.sin(3*beta - (7*xi)/2) - 576*lr**2*sigma**3*np.sin(3*beta - (7*xi)/2) + \
		256*lr*rBar*sigma**3*np.sin(3*beta - (7*xi)/2) - 566*lr**2*sigma**4*np.sin(3*beta - (7*xi)/2) + 77*lr*rBar*sigma**4*np.sin(3*beta - (7*xi)/2) + \
		438*lr**2*sigma**5*np.sin(3*beta - (7*xi)/2) + 2304*lr*rBar*sigma*np.sin(beta - 3*xi) - 6912*lr*rBar*sigma**2*np.sin(beta - 3*xi) - \
		4686*lr**2*sigma**3*np.sin(beta - 3*xi) + 9048*lr*rBar*sigma**3*np.sin(beta - 3*xi) + 276*rBar**2*sigma**3*np.sin(beta - 3*xi) + \
		9372*lr**2*sigma**4*np.sin(beta - 3*xi) - 4440*lr*rBar*sigma**4*np.sin(beta - 3*xi) - 5310*lr**2*sigma**5*np.sin(beta - 3*xi) + 576*lr*rBar*np.sin(beta - (5*xi)/2) - \
		2304*lr*rBar*sigma*np.sin(beta - (5*xi)/2) - 9088*lr**2*sigma**2*np.sin(beta - (5*xi)/2) + 6304*lr*rBar*sigma**2*np.sin(beta - (5*xi)/2) + \
		672*rBar**2*sigma**2*np.sin(beta - (5*xi)/2) + 27264*lr**2*sigma**3*np.sin(beta - (5*xi)/2) - 8000*lr*rBar*sigma**3*np.sin(beta - (5*xi)/2) - \
		672*rBar**2*sigma**3*np.sin(beta - (5*xi)/2) - 32039*lr**2*sigma**4*np.sin(beta - (5*xi)/2) + 3492*lr*rBar*sigma**4*np.sin(beta - (5*xi)/2) + \
		13863*lr**2*sigma**5*np.sin(beta - (5*xi)/2) - 320*lr**2*np.sin(3*beta - (5*xi)/2) - 64*lr*rBar*np.sin(3*beta - (5*xi)/2) + \
		1600*lr**2*sigma*np.sin(3*beta - (5*xi)/2) + 256*lr*rBar*sigma*np.sin(3*beta - (5*xi)/2) - 2560*lr**2*sigma**2*np.sin(3*beta - (5*xi)/2) + \
		3264*lr*rBar*sigma**2*np.sin(3*beta - (5*xi)/2) + 96*rBar**2*sigma**2*np.sin(3*beta - (5*xi)/2) + 1280*lr**2*sigma**3*np.sin(3*beta - (5*xi)/2) - \
		7040*lr*rBar*sigma**3*np.sin(3*beta - (5*xi)/2) - 96*rBar**2*sigma**3*np.sin(3*beta - (5*xi)/2) + 570*lr**2*sigma**4*np.sin(3*beta - (5*xi)/2) + \
		4002*lr*rBar*sigma**4*np.sin(3*beta - (5*xi)/2) - 570*lr**2*sigma**5*np.sin(3*beta - (5*xi)/2) - 7792*lr**2*sigma*np.sin(beta - 2*xi) + \
		64*lr*rBar*sigma*np.sin(beta - 2*xi) + 224*rBar**2*sigma*np.sin(beta - 2*xi) + 31168*lr**2*sigma**2*np.sin(beta - 2*xi) - 192*lr*rBar*sigma**2*np.sin(beta - 2*xi) - \
		448*rBar**2*sigma**2*np.sin(beta - 2*xi) - 59248*lr**2*sigma**3*np.sin(beta - 2*xi) - 640*lr*rBar*sigma**3*np.sin(beta - 2*xi) - 208*rBar**2*sigma**3*np.sin(beta - 2*xi) + \
		56160*lr**2*sigma**4*np.sin(beta - 2*xi) + 768*lr*rBar*sigma**4*np.sin(beta - 2*xi) - 21312*lr**2*sigma**5*np.sin(beta - 2*xi) - 368*lr**2*sigma*np.sin(3*beta - 2*xi) + \
		3520*lr*rBar*sigma*np.sin(3*beta - 2*xi) + 16*rBar**2*sigma*np.sin(3*beta - 2*xi) + 1472*lr**2*sigma**2*np.sin(3*beta - 2*xi) - 10560*lr*rBar*sigma**2*np.sin(3*beta - 2*xi) - \
		32*rBar**2*sigma**2*np.sin(3*beta - 2*xi) - 976*lr**2*sigma**3*np.sin(3*beta - 2*xi) + 11800*lr*rBar*sigma**3*np.sin(3*beta - 2*xi) - \
		152*rBar**2*sigma**3*np.sin(3*beta - 2*xi) - 992*lr**2*sigma**4*np.sin(3*beta - 2*xi) - 4760*lr*rBar*sigma**4*np.sin(3*beta - 2*xi) + \
		960*lr**2*sigma**5*np.sin(3*beta - 2*xi) - 2112*lr**2*np.sin(beta - (3*xi)/2) - 192*lr*rBar*np.sin(beta - (3*xi)/2) + 10560*lr**2*sigma*np.sin(beta - (3*xi)/2) + \
		768*lr*rBar*sigma*np.sin(beta - (3*xi)/2) - 33600*lr**2*sigma**2*np.sin(beta - (3*xi)/2) - 4224*lr*rBar*sigma**2*np.sin(beta - (3*xi)/2) - \
		1440*rBar**2*sigma**2*np.sin(beta - (3*xi)/2) + 58560*lr**2*sigma**3*np.sin(beta - (3*xi)/2) + 6912*lr*rBar*sigma**3*np.sin(beta - (3*xi)/2) + \
		1440*rBar**2*sigma**3*np.sin(beta - (3*xi)/2) - 52758*lr**2*sigma**4*np.sin(beta - (3*xi)/2) - 3132*lr*rBar*sigma**4*np.sin(beta - (3*xi)/2) + \
		19350*lr**2*sigma**5*np.sin(beta - (3*xi)/2) + 1728*lr*rBar*np.sin(3*beta - (3*xi)/2) - 6912*lr*rBar*sigma*np.sin(3*beta - (3*xi)/2) + \
		10656*lr*rBar*sigma**2*np.sin(3*beta - (3*xi)/2) - 288*rBar**2*sigma**2*np.sin(3*beta - (3*xi)/2) - 7488*lr*rBar*sigma**3*np.sin(3*beta - (3*xi)/2) + \
		288*rBar**2*sigma**3*np.sin(3*beta - (3*xi)/2) + 729*lr**2*sigma**4*np.sin(3*beta - (3*xi)/2) + 1962*lr*rBar*sigma**4*np.sin(3*beta - (3*xi)/2) - \
		729*lr**2*sigma**5*np.sin(3*beta - (3*xi)/2) - 4096*lr**2*sigma*np.sin(beta - xi) - 1024*lr*rBar*sigma*np.sin(beta - xi) - 640*rBar**2*sigma*np.sin(beta - xi) + \
		16384*lr**2*sigma**2*np.sin(beta - xi) + 3072*lr*rBar*sigma**2*np.sin(beta - xi) + 1280*rBar**2*sigma**2*np.sin(beta - xi) - \
		30684*lr**2*sigma**3*np.sin(beta - xi) - 2392*lr*rBar*sigma**3*np.sin(beta - xi) - 814*rBar**2*sigma**3*np.sin(beta - xi) + \
		28600*lr**2*sigma**4*np.sin(beta - xi) + 344*lr*rBar*sigma**4*np.sin(beta - xi) - 10716*lr**2*sigma**5*np.sin(beta - xi) - 108*lr**2*sigma**3*np.sin(3*(beta - xi)) + \
		1512*lr*rBar*sigma**3*np.sin(3*(beta - xi)) + 54*rBar**2*sigma**3*np.sin(3*(beta - xi)) + 216*lr**2*sigma**4*np.sin(3*(beta - xi)) - \
		1512*lr*rBar*sigma**4*np.sin(3*(beta - xi)) - 108*lr**2*sigma**5*np.sin(3*(beta - xi)) + 320*lr*rBar*sigma*np.sin(3*beta - xi) + \
		64*rBar**2*sigma*np.sin(3*beta - xi) - 960*lr*rBar*sigma**2*np.sin(3*beta - xi) - 128*rBar**2*sigma**2*np.sin(3*beta - xi) + 138*lr**2*sigma**3*np.sin(3*beta - xi) - \
		200*lr*rBar*sigma**3*np.sin(3*beta - xi) + 340*rBar**2*sigma**3*np.sin(3*beta - xi) - 276*lr**2*sigma**4*np.sin(3*beta - xi) + \
		840*lr*rBar*sigma**4*np.sin(3*beta - xi) + 282*lr**2*sigma**5*np.sin(3*beta - xi) - 1600*lr**2*np.sin(beta - xi/2) - 128*lr*rBar*np.sin(beta - xi/2) + \
		8000*lr**2*sigma*np.sin(beta - xi/2) + 512*lr*rBar*sigma*np.sin(beta - xi/2) - 20480*lr**2*sigma**2*np.sin(beta - xi/2) + 128*lr*rBar*sigma**2*np.sin(beta - xi/2) + \
		192*rBar**2*sigma**2*np.sin(beta - xi/2) + 29440*lr**2*sigma**3*np.sin(beta - xi/2) - 1280*lr*rBar*sigma**3*np.sin(beta - xi/2) - 192*rBar**2*sigma**3*np.sin(beta - xi/2) - \
		22758*lr**2*sigma**4*np.sin(beta - xi/2) + 867*lr*rBar*sigma**4*np.sin(beta - xi/2) + 7398*lr**2*sigma**5*np.sin(beta - xi/2) - 480*lr*rBar*sigma**2*np.sin(3*beta - xi/2) + \
		768*rBar**2*sigma**2*np.sin(3*beta - xi/2) + 960*lr*rBar*sigma**3*np.sin(3*beta - xi/2) - 768*rBar**2*sigma**3*np.sin(3*beta - xi/2) + 45*lr**2*sigma**4*np.sin(3*beta - xi/2) - \
		711*lr*rBar*sigma**4*np.sin(3*beta - xi/2) - 45*lr**2*sigma**5*np.sin(3*beta - xi/2) - 1152*lr*rBar*np.sin(beta + xi/2) + 4608*lr*rBar*sigma*np.sin(beta + xi/2) - \
		9472*lr**2*sigma**2*np.sin(beta + xi/2) - 11840*lr*rBar*sigma**2*np.sin(beta + xi/2) - 192*rBar**2*sigma**2*np.sin(beta + xi/2) + 28416*lr**2*sigma**3*np.sin(beta + xi/2) + \
		14464*lr*rBar*sigma**3*np.sin(beta + xi/2) + 192*rBar**2*sigma**3*np.sin(beta + xi/2) - 31031*lr**2*sigma**4*np.sin(beta + xi/2) - 6797*lr*rBar*sigma**4*np.sin(beta + xi/2) + \
		12087*lr**2*sigma**5*np.sin(beta + xi/2) + 576*lr*rBar*sigma*np.sin(beta + xi) + 192*rBar**2*sigma*np.sin(beta + xi) - 1728*lr*rBar*sigma**2*np.sin(beta + xi) - \
		384*rBar**2*sigma**2*np.sin(beta + xi) - 6726*lr**2*sigma**3*np.sin(beta + xi) + 288*lr*rBar*sigma**3*np.sin(beta + xi) + 192*rBar**2*sigma**3*np.sin(beta + xi) + \
		13452*lr**2*sigma**4*np.sin(beta + xi) + 864*lr*rBar*sigma**4*np.sin(beta + xi) - 7254*lr**2*sigma**5*np.sin(beta + xi) + 6*rBar**2*sigma**3*np.sin(3*beta + xi) + \
		9*lr*rBar*sigma**4*np.sin((6*beta + xi)/2) + 3456*lr*rBar*sigma**2*np.sin(beta + (3*xi)/2) + 576*rBar**2*sigma**2*np.sin(beta + (3*xi)/2) - \
		6912*lr*rBar*sigma**3*np.sin(beta + (3*xi)/2) - 576*rBar**2*sigma**3*np.sin(beta + (3*xi)/2) - 2259*lr**2*sigma**4*np.sin(beta + (3*xi)/2) + \
		3438*lr*rBar*sigma**4*np.sin(beta + (3*xi)/2) + 2259*lr**2*sigma**5*np.sin(beta + (3*xi)/2) - 144*rBar**2*sigma*np.sin(beta + 2*xi) + 288*rBar**2*sigma**2*np.sin(beta + 2*xi) + \
		2136*lr*rBar*sigma**3*np.sin(beta + 2*xi) + 24*rBar**2*sigma**3*np.sin(beta + 2*xi) - 2136*lr*rBar*sigma**4*np.sin(beta + 2*xi) - \
		288*lr**2*sigma**5*np.sin(beta + 2*xi) - 384*rBar**2*sigma**2*np.sin(beta + (5*xi)/2) + 384*rBar**2*sigma**3*np.sin(beta + (5*xi)/2) + \
		390*lr*rBar*sigma**4*np.sin(beta + (5*xi)/2) - 150*rBar**2*sigma**3*np.sin(beta + 3*xi)))/(4*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
		6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - \
		2*rBar*sigma*np.sin(beta + xi/2))))))/(64*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + \
		lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3 \
	)

deriv3NumBoundLambda = lambda sigma,lr,rBar : 98*(lr**4*sigma) + 178*(lr**3*rBar*sigma) + 2126*(lr**4*sigma**2) + 672*(lr**5*sigma**2) + \
4096*(lr**6*sigma**2) + 2286*(lr**3*rBar*sigma**2) + 1920*(lr**4*rBar*sigma**2) + 8192*(lr**5*rBar*sigma**2) + \
281*(lr**2*rBar**2*sigma**2) + 1280*(lr**3*rBar**2*sigma**2) + 4096*(lr**4*rBar**2*sigma**2) + 11636*(lr**4*sigma**3) + \
12684*(lr**5*sigma**3) + 81920*(lr**6*sigma**3) + 15219*(lr**3*rBar*sigma**3) + 28204*(lr**4*rBar*sigma**3) + \
148480*(lr**5*rBar*sigma**3) + 1959*(lr**2*rBar**2*sigma**3) + 14876*(lr**3*rBar**2*sigma**3) + 67776*(lr**4*rBar**2*sigma**3) + \
113*(lr*rBar**3*sigma**3) + 688*(lr**2*rBar**3*sigma**3) + 4096*(lr**3*rBar**3*sigma**3) + 75916*(lr**4*sigma**4) + \
98916*(lr**5*sigma**4) + 736576*(lr**6*sigma**4) + 61982*(lr**3*rBar*sigma**4) + 179928*(lr**4*rBar*sigma**4) + \
1193472*(lr**5*rBar*sigma**4) + 12704*(lr**2*rBar**2*sigma**4) + 75693*(lr**3*rBar**2*sigma**4) + 488032*(lr**4*rBar**2*sigma**4) + \
395*(lr*rBar**3*sigma**4) + 5202*(lr**2*rBar**3*sigma**4) + 46944*(lr**3*rBar**3*sigma**4) + 7*(rBar**4*sigma**4) + \
106*(lr*rBar**4*sigma**4) + 1344*(lr**2*rBar**4*sigma**4) + 521387*(lr**4*sigma**5) + 474907*(lr**5*sigma**5) + \
3921040*(lr**6*sigma**5) + 277825*(lr**3*rBar*sigma**5) + 717807*(lr**4*rBar*sigma**5) + 5584480*(lr**5*rBar*sigma**5) + \
22868*(lr**2*rBar**2*sigma**5) + 204343*(lr**3*rBar**2*sigma**5) + 1997904*(lr**4*rBar**2*sigma**5) + 1436*(lr*rBar**3*sigma**5) + \
17568*(lr**2*rBar**3*sigma**5) + 224416*(lr**3*rBar**3*sigma**5) + 403*(lr*rBar**4*sigma**5) + 10192*(lr**2*rBar**4*sigma**5) + \
5*(rBar**5*sigma**5) + 176*(lr*rBar**5*sigma**5) + 1226130*(lr**4*sigma**6) + 1555082*(lr**5*sigma**6) + 13685720*(lr**6*sigma**6) + \
424720*(lr**3*rBar*sigma**6) + 1738987*(lr**4*rBar*sigma**6) + 16766040*(lr**5*rBar*sigma**6) + 38320*(lr**2*rBar**2*sigma**6) + \
424426*(lr**3*rBar**2*sigma**6) + 5086496*(lr**4*rBar**2*sigma**6) + 22700*(lr**2*rBar**3*sigma**6) + 571992*(lr**3*rBar**3*sigma**6) \
+ 560*(lr*rBar**4*sigma**6) + 28936*(lr**2*rBar**4*sigma**6) + 672*(lr*rBar**5*sigma**6) + 8*(rBar**6*sigma**6) + \
836338*(lr**4*sigma**7) + 2776004*(lr**5*sigma**7) + 32726752*(lr**6*sigma**7) + 225243*(lr**3*rBar*sigma**7) + \
2927964*(lr**4*rBar*sigma**7) + 33496304*(lr**5*rBar*sigma**7) + 404970*(lr**3*rBar**2*sigma**7) + 8270658*(lr**4*rBar**2*sigma**7) + \
18571*(lr**2*rBar**3*sigma**7) + 820534*(lr**3*rBar**3*sigma**7) + 36448*(lr**2*rBar**4*sigma**7) + 640*(lr*rBar**5*sigma**7) + \
331641*(lr**4*sigma**8) + 3230633*(lr**5*sigma**8) + 54301600*(lr**6*sigma**8) + 1935097*(lr**4*rBar*sigma**8) + \
44537280*(lr**5*rBar*sigma**8) + 192936*(lr**3*rBar**2*sigma**8) + 8365734*(lr**4*rBar**2*sigma**8) + 634394*(lr**3*rBar**3*sigma**8) \
+ 17540*(lr**2*rBar**4*sigma**8) + 1784952*(lr**5*sigma**9) + 61732352*(lr**6*sigma**9) + 698034*(lr**4*rBar*sigma**9) + \
38006144*(lr**5*rBar*sigma**9) + 4901151*(lr**4*rBar**2*sigma**9) + 205969*(lr**3*rBar**3*sigma**9) + 511650*(lr**5*sigma**10) + \
46019200*(lr**6*sigma**10) + 18889600*(lr**5*rBar*sigma**10) + 1238904*(lr**4*rBar**2*sigma**10) + 20313600*(lr**6*sigma**11) + \
4166400*(lr**5*rBar*sigma**11) + 4032000*(lr**6*sigma**12)

deriv3NonProportionalDenoms = [0, 9, 15, 18]

deriv3DenomLambdas = (\
	lambda xi,beta,sigma,lr,rBar : (3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - \
	(3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - \
	2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + \
	lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 2*(3*lr*sigma**2*np.sin(beta) + \
	lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) \
	+ 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 4*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
	6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
	sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 8*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + \
	4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta \
	+ xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - \
	xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 16*(3*lr*sigma**2*np.sin(beta) + \
	lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) \
	+ 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 32*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
	6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
	sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 64*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + \
	4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta \
	+ xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - \
	xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 128*(3*lr*sigma**2*np.sin(beta) + \
	lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) \
	+ 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 256*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
	6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
	sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : (3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + \
	4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta \
	+ xi/2))**2*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta \
	- xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 2*(3*lr*sigma**2*np.sin(beta) + \
	lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) \
	+ 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))**2*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 4*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
	6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
	sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))**2*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 8*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + \
	4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta \
	+ xi/2))**2*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta \
	- xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 16*(3*lr*sigma**2*np.sin(beta) + \
	lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) \
	+ 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))**2*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 32*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
	6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
	sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))**2*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : (3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + \
	4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta \
	+ xi/2))**3*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta \
	- xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 2*(3*lr*sigma**2*np.sin(beta) + \
	lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) \
	+ 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))**3*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 4*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
	6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
	sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))**3*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : (-2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) + rBar*sigma*np.cos(beta)*np.sin(xi/2) - \
	lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) \
	- rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 2*(-2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) + rBar*sigma*np.cos(beta)*np.sin(xi/2) - \
	lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) \
	- rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 4*(-2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) + rBar*sigma*np.cos(beta)*np.sin(xi/2) - \
	lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) \
	- rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 8*(-2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) + rBar*sigma*np.cos(beta)*np.sin(xi/2) - \
	lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) \
	- rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 16*(-2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) + rBar*sigma*np.cos(beta)*np.sin(xi/2) - \
	lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) \
	- rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3 \
)

deriv3DenomLambdas = [ deriv3DenomLambdas[k] for k in deriv3NonProportionalDenoms]

deriv3DenomBoundsLambdas = (\
	lambda sigma,lr,rBar : np.sqrt(131072*(lr)**8 + 4456448*(lr**8*sigma) + 483328*(lr**7*rBar*sigma) + 71037952*(lr**8*sigma**2) + \
	14295040*(lr**7*rBar*sigma**2) + 782080*(lr**6*rBar**2*sigma**2) + 704709632*(lr**8*sigma**3) + 196466944*(lr**7*rBar*sigma**3) + \
	19737600*(lr**6*rBar**2*sigma**3) + 726272*(lr**5*rBar**3*sigma**3) + 4869364544*(lr**8*sigma**4) + 1662951552*(lr**7*rBar*sigma**4) + \
	228665088*(lr**6*rBar**2*sigma**4) + 15234560*(lr**5*rBar**3*sigma**4) + 424000*(lr**4*rBar**4*sigma**4) + \
	24849784704*(lr**8*sigma**5) + 9683804928*(lr**7*rBar*sigma**5) + 1607803200*(lr**6*rBar**2*sigma**5) + \
	144137728*(lr**5*rBar**3*sigma**5) + 7114240*(lr**4*rBar**4*sigma**5) + 159616*(lr**3*rBar**5*sigma**5) + 96885695072*(lr**8*sigma**6) \
	+ 41037340128*(lr**7*rBar*sigma**6) + 7640245968*(lr**6*rBar**2*sigma**6) + 809777024*(lr**5*rBar**3*sigma**6) + \
	52384576*(lr**4*rBar**4*sigma**6) + 2014464*(lr**3*rBar**5*sigma**6) + 37904*(lr**2*rBar**6*sigma**6) + 294381962528*(lr**8*sigma**7) \
	+ 130500123280*(lr**7*rBar*sigma**7) + 25845371808*(lr**6*rBar**2*sigma**7) + 2990706704*(lr**5*rBar**3*sigma**7) + \
	220985856*(lr**4*rBar**4*sigma**7) + 10632528*(lr**3*rBar**5*sigma**7) + 320960*(lr**2*rBar**6*sigma**7) + 5200*(lr*rBar**7*sigma**7) \
	+ 704475796036*(lr**8*sigma**8) + 316337408848*(lr**7*rBar*sigma**8) + 63808283968*(lr**6*rBar**2*sigma**8) + \
	7584777168*(lr**5*rBar**3*sigma**8) + 583848552*(lr**4*rBar**4*sigma**8) + 30015384*(lr**3*rBar**5*sigma**8) + \
	1022944*(lr**2*rBar**6*sigma**8) + 22240*(lr*rBar**7*sigma**8) + 316*(rBar**8*sigma**8) + 1332197931584*(lr**8*sigma**9) + \
	587334747088*(lr**7*rBar*sigma**9) + 115825266624*(lr**6*rBar**2*sigma**9) + 13373306008*(lr**5*rBar**3*sigma**9) + \
	988741776*(lr**4*rBar**4*sigma**9) + 47754264*(lr**3*rBar**5*sigma**9) + 1451996*(lr**2*rBar**6*sigma**9) + \
	23776*(lr*rBar**7*sigma**9) + 1984157440448*(lr**8*sigma**10) + 831144788448*(lr**7*rBar*sigma**10) + \
	153399703812*(lr**6*rBar**2*sigma**10) + 16183024348*(lr**5*rBar**3*sigma**10) + 1047678620*(lr**4*rBar**4*sigma**10) + \
	40569804*(lr**3*rBar**5*sigma**10) + 774584*(lr**2*rBar**6*sigma**10) + 2303035589376*(lr**8*sigma**11) + \
	882400410816*(lr**7*rBar*sigma**11) + 144546255456*(lr**6*rBar**2*sigma**11) + 12860211530*(lr**5*rBar**3*sigma**11) + \
	634851899*(lr**4*rBar**4*sigma**11) + 14385956*(lr**3*rBar**5*sigma**11) + 2042281601408*(lr**8*sigma**12) + \
	681517613136*(lr**7*rBar*sigma**12) + 91977843504*(lr**6*rBar**2*sigma**12) + 6059286268*(lr**5*rBar**3*sigma**12) + \
	168462959*(lr**4*rBar**4*sigma**12) + 1337591708672*(lr**8*sigma**13) + 361976983840*(lr**7*rBar*sigma**13) + \
	35484422136*(lr**6*rBar**2*sigma**13) + 1285399660*(lr**5*rBar**3*sigma**13) + 610210004992*(lr**8*sigma**14) + \
	118344729280*(lr**7*rBar*sigma**14) + 6276608548*(lr**6*rBar**2*sigma**14) + 173244531968*(lr**8*sigma**15) + \
	17968872520*(lr**7*rBar*sigma**15) + 23060538368*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : np.sqrt(524288*(lr)**8 + \
	17825792*(lr**8*sigma) + 1933312*(lr**7*rBar*sigma) + 284151808*(lr**8*sigma**2) + 57180160*(lr**7*rBar*sigma**2) + \
	3128320*(lr**6*rBar**2*sigma**2) + 2818838528*(lr**8*sigma**3) + 785867776*(lr**7*rBar*sigma**3) + 78950400*(lr**6*rBar**2*sigma**3) + \
	2905088*(lr**5*rBar**3*sigma**3) + 19477458176*(lr**8*sigma**4) + 6651806208*(lr**7*rBar*sigma**4) + \
	914660352*(lr**6*rBar**2*sigma**4) + 60938240*(lr**5*rBar**3*sigma**4) + 1696000*(lr**4*rBar**4*sigma**4) + \
	99399138816*(lr**8*sigma**5) + 38735219712*(lr**7*rBar*sigma**5) + 6431212800*(lr**6*rBar**2*sigma**5) + \
	576550912*(lr**5*rBar**3*sigma**5) + 28456960*(lr**4*rBar**4*sigma**5) + 638464*(lr**3*rBar**5*sigma**5) + \
	387542780288*(lr**8*sigma**6) + 164149360512*(lr**7*rBar*sigma**6) + 30560983872*(lr**6*rBar**2*sigma**6) + \
	3239108096*(lr**5*rBar**3*sigma**6) + 209538304*(lr**4*rBar**4*sigma**6) + 8057856*(lr**3*rBar**5*sigma**6) + \
	151616*(lr**2*rBar**6*sigma**6) + 1177527850112*(lr**8*sigma**7) + 522000493120*(lr**7*rBar*sigma**7) + \
	103381487232*(lr**6*rBar**2*sigma**7) + 11962826816*(lr**5*rBar**3*sigma**7) + 883943424*(lr**4*rBar**4*sigma**7) + \
	42530112*(lr**3*rBar**5*sigma**7) + 1283840*(lr**2*rBar**6*sigma**7) + 20800*(lr*rBar**7*sigma**7) + 2817903184144*(lr**8*sigma**8) + \
	1265349635392*(lr**7*rBar*sigma**8) + 255233135872*(lr**6*rBar**2*sigma**8) + 30339108672*(lr**5*rBar**3*sigma**8) + \
	2335394208*(lr**4*rBar**4*sigma**8) + 120061536*(lr**3*rBar**5*sigma**8) + 4091776*(lr**2*rBar**6*sigma**8) + \
	88960*(lr*rBar**7*sigma**8) + 1264*(rBar**8*sigma**8) + 5328791726336*(lr**8*sigma**9) + 2349338988352*(lr**7*rBar*sigma**9) + \
	463301066496*(lr**6*rBar**2*sigma**9) + 53493224032*(lr**5*rBar**3*sigma**9) + 3954967104*(lr**4*rBar**4*sigma**9) + \
	191017056*(lr**3*rBar**5*sigma**9) + 5807984*(lr**2*rBar**6*sigma**9) + 95104*(lr*rBar**7*sigma**9) + 7936629761792*(lr**8*sigma**10) \
	+ 3324579153792*(lr**7*rBar*sigma**10) + 613598815248*(lr**6*rBar**2*sigma**10) + 64732097392*(lr**5*rBar**3*sigma**10) + \
	4190714480*(lr**4*rBar**4*sigma**10) + 162279216*(lr**3*rBar**5*sigma**10) + 3098336*(lr**2*rBar**6*sigma**10) + \
	9212142357504*(lr**8*sigma**11) + 3529601643264*(lr**7*rBar*sigma**11) + 578185021824*(lr**6*rBar**2*sigma**11) + \
	51440846120*(lr**5*rBar**3*sigma**11) + 2539407596*(lr**4*rBar**4*sigma**11) + 57543824*(lr**3*rBar**5*sigma**11) + \
	8169126405632*(lr**8*sigma**12) + 2726070452544*(lr**7*rBar*sigma**12) + 367911374016*(lr**6*rBar**2*sigma**12) + \
	24237145072*(lr**5*rBar**3*sigma**12) + 673851836*(lr**4*rBar**4*sigma**12) + 5350366834688*(lr**8*sigma**13) + \
	1447907935360*(lr**7*rBar*sigma**13) + 141937688544*(lr**6*rBar**2*sigma**13) + 5141598640*(lr**5*rBar**3*sigma**13) + \
	2440840019968*(lr**8*sigma**14) + 473378917120*(lr**7*rBar*sigma**14) + 25106434192*(lr**6*rBar**2*sigma**14) + \
	692978127872*(lr**8*sigma**15) + 71875490080*(lr**7*rBar*sigma**15) + 92242153472*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : \
	np.sqrt(2097152*(lr)**8 + 71303168*(lr**8*sigma) + 7733248*(lr**7*rBar*sigma) + 1136607232*(lr**8*sigma**2) + \
	228720640*(lr**7*rBar*sigma**2) + 12513280*(lr**6*rBar**2*sigma**2) + 11275354112*(lr**8*sigma**3) + 3143471104*(lr**7*rBar*sigma**3) \
	+ 315801600*(lr**6*rBar**2*sigma**3) + 11620352*(lr**5*rBar**3*sigma**3) + 77909832704*(lr**8*sigma**4) + \
	26607224832*(lr**7*rBar*sigma**4) + 3658641408*(lr**6*rBar**2*sigma**4) + 243752960*(lr**5*rBar**3*sigma**4) + \
	6784000*(lr**4*rBar**4*sigma**4) + 397596555264*(lr**8*sigma**5) + 154940878848*(lr**7*rBar*sigma**5) + \
	25724851200*(lr**6*rBar**2*sigma**5) + 2306203648*(lr**5*rBar**3*sigma**5) + 113827840*(lr**4*rBar**4*sigma**5) + \
	2553856*(lr**3*rBar**5*sigma**5) + 1550171121152*(lr**8*sigma**6) + 656597442048*(lr**7*rBar*sigma**6) + \
	122243935488*(lr**6*rBar**2*sigma**6) + 12956432384*(lr**5*rBar**3*sigma**6) + 838153216*(lr**4*rBar**4*sigma**6) + \
	32231424*(lr**3*rBar**5*sigma**6) + 606464*(lr**2*rBar**6*sigma**6) + 4710111400448*(lr**8*sigma**7) + \
	2088001972480*(lr**7*rBar*sigma**7) + 413525948928*(lr**6*rBar**2*sigma**7) + 47851307264*(lr**5*rBar**3*sigma**7) + \
	3535773696*(lr**4*rBar**4*sigma**7) + 170120448*(lr**3*rBar**5*sigma**7) + 5135360*(lr**2*rBar**6*sigma**7) + \
	83200*(lr*rBar**7*sigma**7) + 11271612736576*(lr**8*sigma**8) + 5061398541568*(lr**7*rBar*sigma**8) + \
	1020932543488*(lr**6*rBar**2*sigma**8) + 121356434688*(lr**5*rBar**3*sigma**8) + 9341576832*(lr**4*rBar**4*sigma**8) + \
	480246144*(lr**3*rBar**5*sigma**8) + 16367104*(lr**2*rBar**6*sigma**8) + 355840*(lr*rBar**7*sigma**8) + 5056*(rBar**8*sigma**8) + \
	21315166905344*(lr**8*sigma**9) + 9397355953408*(lr**7*rBar*sigma**9) + 1853204265984*(lr**6*rBar**2*sigma**9) + \
	213972896128*(lr**5*rBar**3*sigma**9) + 15819868416*(lr**4*rBar**4*sigma**9) + 764068224*(lr**3*rBar**5*sigma**9) + \
	23231936*(lr**2*rBar**6*sigma**9) + 380416*(lr*rBar**7*sigma**9) + 31746519047168*(lr**8*sigma**10) + \
	13298316615168*(lr**7*rBar*sigma**10) + 2454395260992*(lr**6*rBar**2*sigma**10) + 258928389568*(lr**5*rBar**3*sigma**10) + \
	16762857920*(lr**4*rBar**4*sigma**10) + 649116864*(lr**3*rBar**5*sigma**10) + 12393344*(lr**2*rBar**6*sigma**10) + \
	36848569430016*(lr**8*sigma**11) + 14118406573056*(lr**7*rBar*sigma**11) + 2312740087296*(lr**6*rBar**2*sigma**11) + \
	205763384480*(lr**5*rBar**3*sigma**11) + 10157630384*(lr**4*rBar**4*sigma**11) + 230175296*(lr**3*rBar**5*sigma**11) + \
	32676505622528*(lr**8*sigma**12) + 10904281810176*(lr**7*rBar*sigma**12) + 1471645496064*(lr**6*rBar**2*sigma**12) + \
	96948580288*(lr**5*rBar**3*sigma**12) + 2695407344*(lr**4*rBar**4*sigma**12) + 21401467338752*(lr**8*sigma**13) + \
	5791631741440*(lr**7*rBar*sigma**13) + 567750754176*(lr**6*rBar**2*sigma**13) + 20566394560*(lr**5*rBar**3*sigma**13) + \
	9763360079872*(lr**8*sigma**14) + 1893515668480*(lr**7*rBar*sigma**14) + 100425736768*(lr**6*rBar**2*sigma**14) + \
	2771912511488*(lr**8*sigma**15) + 287501960320*(lr**7*rBar*sigma**15) + 368968613888*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : \
	np.sqrt(8388608*(lr)**8 + 285212672*(lr**8*sigma) + 30932992*(lr**7*rBar*sigma) + 4546428928*(lr**8*sigma**2) + \
	914882560*(lr**7*rBar*sigma**2) + 50053120*(lr**6*rBar**2*sigma**2) + 45101416448*(lr**8*sigma**3) + 12573884416*(lr**7*rBar*sigma**3) \
	+ 1263206400*(lr**6*rBar**2*sigma**3) + 46481408*(lr**5*rBar**3*sigma**3) + 311639330816*(lr**8*sigma**4) + \
	106428899328*(lr**7*rBar*sigma**4) + 14634565632*(lr**6*rBar**2*sigma**4) + 975011840*(lr**5*rBar**3*sigma**4) + \
	27136000*(lr**4*rBar**4*sigma**4) + 1590386221056*(lr**8*sigma**5) + 619763515392*(lr**7*rBar*sigma**5) + \
	102899404800*(lr**6*rBar**2*sigma**5) + 9224814592*(lr**5*rBar**3*sigma**5) + 455311360*(lr**4*rBar**4*sigma**5) + \
	10215424*(lr**3*rBar**5*sigma**5) + 6200684484608*(lr**8*sigma**6) + 2626389768192*(lr**7*rBar*sigma**6) + \
	488975741952*(lr**6*rBar**2*sigma**6) + 51825729536*(lr**5*rBar**3*sigma**6) + 3352612864*(lr**4*rBar**4*sigma**6) + \
	128925696*(lr**3*rBar**5*sigma**6) + 2425856*(lr**2*rBar**6*sigma**6) + 18840445601792*(lr**8*sigma**7) + \
	8352007889920*(lr**7*rBar*sigma**7) + 1654103795712*(lr**6*rBar**2*sigma**7) + 191405229056*(lr**5*rBar**3*sigma**7) + \
	14143094784*(lr**4*rBar**4*sigma**7) + 680481792*(lr**3*rBar**5*sigma**7) + 20541440*(lr**2*rBar**6*sigma**7) + \
	332800*(lr*rBar**7*sigma**7) + 45086450946304*(lr**8*sigma**8) + 20245594166272*(lr**7*rBar*sigma**8) + \
	4083730173952*(lr**6*rBar**2*sigma**8) + 485425738752*(lr**5*rBar**3*sigma**8) + 37366307328*(lr**4*rBar**4*sigma**8) + \
	1920984576*(lr**3*rBar**5*sigma**8) + 65468416*(lr**2*rBar**6*sigma**8) + 1423360*(lr*rBar**7*sigma**8) + 20224*(rBar**8*sigma**8) + \
	85260667621376*(lr**8*sigma**9) + 37589423813632*(lr**7*rBar*sigma**9) + 7412817063936*(lr**6*rBar**2*sigma**9) + \
	855891584512*(lr**5*rBar**3*sigma**9) + 63279473664*(lr**4*rBar**4*sigma**9) + 3056272896*(lr**3*rBar**5*sigma**9) + \
	92927744*(lr**2*rBar**6*sigma**9) + 1521664*(lr*rBar**7*sigma**9) + 126986076188672*(lr**8*sigma**10) + \
	53193266460672*(lr**7*rBar*sigma**10) + 9817581043968*(lr**6*rBar**2*sigma**10) + 1035713558272*(lr**5*rBar**3*sigma**10) + \
	67051431680*(lr**4*rBar**4*sigma**10) + 2596467456*(lr**3*rBar**5*sigma**10) + 49573376*(lr**2*rBar**6*sigma**10) + \
	147394277720064*(lr**8*sigma**11) + 56473626292224*(lr**7*rBar*sigma**11) + 9250960349184*(lr**6*rBar**2*sigma**11) + \
	823053537920*(lr**5*rBar**3*sigma**11) + 40630521536*(lr**4*rBar**4*sigma**11) + 920701184*(lr**3*rBar**5*sigma**11) + \
	130706022490112*(lr**8*sigma**12) + 43617127240704*(lr**7*rBar*sigma**12) + 5886581984256*(lr**6*rBar**2*sigma**12) + \
	387794321152*(lr**5*rBar**3*sigma**12) + 10781629376*(lr**4*rBar**4*sigma**12) + 85605869355008*(lr**8*sigma**13) + \
	23166526965760*(lr**7*rBar*sigma**13) + 2271003016704*(lr**6*rBar**2*sigma**13) + 82265578240*(lr**5*rBar**3*sigma**13) + \
	39053440319488*(lr**8*sigma**14) + 7574062673920*(lr**7*rBar*sigma**14) + 401702947072*(lr**6*rBar**2*sigma**14) + \
	11087650045952*(lr**8*sigma**15) + 1150007841280*(lr**7*rBar*sigma**15) + 1475874455552*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : \
	np.sqrt(33554432*(lr)**8 + 1140850688*(lr**8*sigma) + 123731968*(lr**7*rBar*sigma) + 18185715712*(lr**8*sigma**2) + \
	3659530240*(lr**7*rBar*sigma**2) + 200212480*(lr**6*rBar**2*sigma**2) + 180405665792*(lr**8*sigma**3) + \
	50295537664*(lr**7*rBar*sigma**3) + 5052825600*(lr**6*rBar**2*sigma**3) + 185925632*(lr**5*rBar**3*sigma**3) + \
	1246557323264*(lr**8*sigma**4) + 425715597312*(lr**7*rBar*sigma**4) + 58538262528*(lr**6*rBar**2*sigma**4) + \
	3900047360*(lr**5*rBar**3*sigma**4) + 108544000*(lr**4*rBar**4*sigma**4) + 6361544884224*(lr**8*sigma**5) + \
	2479054061568*(lr**7*rBar*sigma**5) + 411597619200*(lr**6*rBar**2*sigma**5) + 36899258368*(lr**5*rBar**3*sigma**5) + \
	1821245440*(lr**4*rBar**4*sigma**5) + 40861696*(lr**3*rBar**5*sigma**5) + 24802737938432*(lr**8*sigma**6) + \
	10505559072768*(lr**7*rBar*sigma**6) + 1955902967808*(lr**6*rBar**2*sigma**6) + 207302918144*(lr**5*rBar**3*sigma**6) + \
	13410451456*(lr**4*rBar**4*sigma**6) + 515702784*(lr**3*rBar**5*sigma**6) + 9703424*(lr**2*rBar**6*sigma**6) + \
	75361782407168*(lr**8*sigma**7) + 33408031559680*(lr**7*rBar*sigma**7) + 6616415182848*(lr**6*rBar**2*sigma**7) + \
	765620916224*(lr**5*rBar**3*sigma**7) + 56572379136*(lr**4*rBar**4*sigma**7) + 2721927168*(lr**3*rBar**5*sigma**7) + \
	82165760*(lr**2*rBar**6*sigma**7) + 1331200*(lr*rBar**7*sigma**7) + 180345803785216*(lr**8*sigma**8) + \
	80982376665088*(lr**7*rBar*sigma**8) + 16334920695808*(lr**6*rBar**2*sigma**8) + 1941702955008*(lr**5*rBar**3*sigma**8) + \
	149465229312*(lr**4*rBar**4*sigma**8) + 7683938304*(lr**3*rBar**5*sigma**8) + 261873664*(lr**2*rBar**6*sigma**8) + \
	5693440*(lr*rBar**7*sigma**8) + 80896*(rBar**8*sigma**8) + 341042670485504*(lr**8*sigma**9) + 150357695254528*(lr**7*rBar*sigma**9) + \
	29651268255744*(lr**6*rBar**2*sigma**9) + 3423566338048*(lr**5*rBar**3*sigma**9) + 253117894656*(lr**4*rBar**4*sigma**9) + \
	12225091584*(lr**3*rBar**5*sigma**9) + 371710976*(lr**2*rBar**6*sigma**9) + 6086656*(lr*rBar**7*sigma**9) + \
	507944304754688*(lr**8*sigma**10) + 212773065842688*(lr**7*rBar*sigma**10) + 39270324175872*(lr**6*rBar**2*sigma**10) + \
	4142854233088*(lr**5*rBar**3*sigma**10) + 268205726720*(lr**4*rBar**4*sigma**10) + 10385869824*(lr**3*rBar**5*sigma**10) + \
	198293504*(lr**2*rBar**6*sigma**10) + 589577110880256*(lr**8*sigma**11) + 225894505168896*(lr**7*rBar*sigma**11) + \
	37003841396736*(lr**6*rBar**2*sigma**11) + 3292214151680*(lr**5*rBar**3*sigma**11) + 162522086144*(lr**4*rBar**4*sigma**11) + \
	3682804736*(lr**3*rBar**5*sigma**11) + 522824089960448*(lr**8*sigma**12) + 174468508962816*(lr**7*rBar*sigma**12) + \
	23546327937024*(lr**6*rBar**2*sigma**12) + 1551177284608*(lr**5*rBar**3*sigma**12) + 43126517504*(lr**4*rBar**4*sigma**12) + \
	342423477420032*(lr**8*sigma**13) + 92666107863040*(lr**7*rBar*sigma**13) + 9084012066816*(lr**6*rBar**2*sigma**13) + \
	329062312960*(lr**5*rBar**3*sigma**13) + 156213761277952*(lr**8*sigma**14) + 30296250695680*(lr**7*rBar*sigma**14) + \
	1606811788288*(lr**6*rBar**2*sigma**14) + 44350600183808*(lr**8*sigma**15) + 4600031365120*(lr**7*rBar*sigma**15) + \
	5903497822208*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : np.sqrt(134217728*(lr)**8 + 4563402752*(lr**8*sigma) + \
	494927872*(lr**7*rBar*sigma) + 72742862848*(lr**8*sigma**2) + 14638120960*(lr**7*rBar*sigma**2) + 800849920*(lr**6*rBar**2*sigma**2) + \
	721622663168*(lr**8*sigma**3) + 201182150656*(lr**7*rBar*sigma**3) + 20211302400*(lr**6*rBar**2*sigma**3) + \
	743702528*(lr**5*rBar**3*sigma**3) + 4986229293056*(lr**8*sigma**4) + 1702862389248*(lr**7*rBar*sigma**4) + \
	234153050112*(lr**6*rBar**2*sigma**4) + 15600189440*(lr**5*rBar**3*sigma**4) + 434176000*(lr**4*rBar**4*sigma**4) + \
	25446179536896*(lr**8*sigma**5) + 9916216246272*(lr**7*rBar*sigma**5) + 1646390476800*(lr**6*rBar**2*sigma**5) + \
	147597033472*(lr**5*rBar**3*sigma**5) + 7284981760*(lr**4*rBar**4*sigma**5) + 163446784*(lr**3*rBar**5*sigma**5) + \
	99210951753728*(lr**8*sigma**6) + 42022236291072*(lr**7*rBar*sigma**6) + 7823611871232*(lr**6*rBar**2*sigma**6) + \
	829211672576*(lr**5*rBar**3*sigma**6) + 53641805824*(lr**4*rBar**4*sigma**6) + 2062811136*(lr**3*rBar**5*sigma**6) + \
	38813696*(lr**2*rBar**6*sigma**6) + 301447129628672*(lr**8*sigma**7) + 133632126238720*(lr**7*rBar*sigma**7) + \
	26465660731392*(lr**6*rBar**2*sigma**7) + 3062483664896*(lr**5*rBar**3*sigma**7) + 226289516544*(lr**4*rBar**4*sigma**7) + \
	10887708672*(lr**3*rBar**5*sigma**7) + 328663040*(lr**2*rBar**6*sigma**7) + 5324800*(lr*rBar**7*sigma**7) + \
	721383215140864*(lr**8*sigma**8) + 323929506660352*(lr**7*rBar*sigma**8) + 65339682783232*(lr**6*rBar**2*sigma**8) + \
	7766811820032*(lr**5*rBar**3*sigma**8) + 597860917248*(lr**4*rBar**4*sigma**8) + 30735753216*(lr**3*rBar**5*sigma**8) + \
	1047494656*(lr**2*rBar**6*sigma**8) + 22773760*(lr*rBar**7*sigma**8) + 323584*(rBar**8*sigma**8) + 1364170681942016*(lr**8*sigma**9) + \
	601430781018112*(lr**7*rBar*sigma**9) + 118605073022976*(lr**6*rBar**2*sigma**9) + 13694265352192*(lr**5*rBar**3*sigma**9) + \
	1012471578624*(lr**4*rBar**4*sigma**9) + 48900366336*(lr**3*rBar**5*sigma**9) + 1486843904*(lr**2*rBar**6*sigma**9) + \
	24346624*(lr*rBar**7*sigma**9) + 2031777219018752*(lr**8*sigma**10) + 851092263370752*(lr**7*rBar*sigma**10) + \
	157081296703488*(lr**6*rBar**2*sigma**10) + 16571416932352*(lr**5*rBar**3*sigma**10) + 1072822906880*(lr**4*rBar**4*sigma**10) + \
	41543479296*(lr**3*rBar**5*sigma**10) + 793174016*(lr**2*rBar**6*sigma**10) + 2358308443521024*(lr**8*sigma**11) + \
	903578020675584*(lr**7*rBar*sigma**11) + 148015365586944*(lr**6*rBar**2*sigma**11) + 13168856606720*(lr**5*rBar**3*sigma**11) + \
	650088344576*(lr**4*rBar**4*sigma**11) + 14731218944*(lr**3*rBar**5*sigma**11) + 2091296359841792*(lr**8*sigma**12) + \
	697874035851264*(lr**7*rBar*sigma**12) + 94185311748096*(lr**6*rBar**2*sigma**12) + 6204709138432*(lr**5*rBar**3*sigma**12) + \
	172506070016*(lr**4*rBar**4*sigma**12) + 1369693909680128*(lr**8*sigma**13) + 370664431452160*(lr**7*rBar*sigma**13) + \
	36336048267264*(lr**6*rBar**2*sigma**13) + 1316249251840*(lr**5*rBar**3*sigma**13) + 624855045111808*(lr**8*sigma**14) + \
	121185002782720*(lr**7*rBar*sigma**14) + 6427247153152*(lr**6*rBar**2*sigma**14) + 177402400735232*(lr**8*sigma**15) + \
	18400125460480*(lr**7*rBar*sigma**15) + 23613991288832*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : np.sqrt(536870912*(lr)**8 + \
	18253611008*(lr**8*sigma) + 1979711488*(lr**7*rBar*sigma) + 290971451392*(lr**8*sigma**2) + 58552483840*(lr**7*rBar*sigma**2) + \
	3203399680*(lr**6*rBar**2*sigma**2) + 2886490652672*(lr**8*sigma**3) + 804728602624*(lr**7*rBar*sigma**3) + \
	80845209600*(lr**6*rBar**2*sigma**3) + 2974810112*(lr**5*rBar**3*sigma**3) + 19944917172224*(lr**8*sigma**4) + \
	6811449556992*(lr**7*rBar*sigma**4) + 936612200448*(lr**6*rBar**2*sigma**4) + 62400757760*(lr**5*rBar**3*sigma**4) + \
	1736704000*(lr**4*rBar**4*sigma**4) + 101784718147584*(lr**8*sigma**5) + 39664864985088*(lr**7*rBar*sigma**5) + \
	6585561907200*(lr**6*rBar**2*sigma**5) + 590388133888*(lr**5*rBar**3*sigma**5) + 29139927040*(lr**4*rBar**4*sigma**5) + \
	653787136*(lr**3*rBar**5*sigma**5) + 396843807014912*(lr**8*sigma**6) + 168088945164288*(lr**7*rBar*sigma**6) + \
	31294447484928*(lr**6*rBar**2*sigma**6) + 3316846690304*(lr**5*rBar**3*sigma**6) + 214567223296*(lr**4*rBar**4*sigma**6) + \
	8251244544*(lr**3*rBar**5*sigma**6) + 155254784*(lr**2*rBar**6*sigma**6) + 1205788518514688*(lr**8*sigma**7) + \
	534528504954880*(lr**7*rBar*sigma**7) + 105862642925568*(lr**6*rBar**2*sigma**7) + 12249934659584*(lr**5*rBar**3*sigma**7) + \
	905158066176*(lr**4*rBar**4*sigma**7) + 43550834688*(lr**3*rBar**5*sigma**7) + 1314652160*(lr**2*rBar**6*sigma**7) + \
	21299200*(lr*rBar**7*sigma**7) + 2885532860563456*(lr**8*sigma**8) + 1295718026641408*(lr**7*rBar*sigma**8) + \
	261358731132928*(lr**6*rBar**2*sigma**8) + 31067247280128*(lr**5*rBar**3*sigma**8) + 2391443668992*(lr**4*rBar**4*sigma**8) + \
	122943012864*(lr**3*rBar**5*sigma**8) + 4189978624*(lr**2*rBar**6*sigma**8) + 91095040*(lr*rBar**7*sigma**8) + \
	1294336*(rBar**8*sigma**8) + 5456682727768064*(lr**8*sigma**9) + 2405723124072448*(lr**7*rBar*sigma**9) + \
	474420292091904*(lr**6*rBar**2*sigma**9) + 54777061408768*(lr**5*rBar**3*sigma**9) + 4049886314496*(lr**4*rBar**4*sigma**9) + \
	195601465344*(lr**3*rBar**5*sigma**9) + 5947375616*(lr**2*rBar**6*sigma**9) + 97386496*(lr*rBar**7*sigma**9) + \
	8127108876075008*(lr**8*sigma**10) + 3404369053483008*(lr**7*rBar*sigma**10) + 628325186813952*(lr**6*rBar**2*sigma**10) + \
	66285667729408*(lr**5*rBar**3*sigma**10) + 4291291627520*(lr**4*rBar**4*sigma**10) + 166173917184*(lr**3*rBar**5*sigma**10) + \
	3172696064*(lr**2*rBar**6*sigma**10) + 9433233774084096*(lr**8*sigma**11) + 3614312082702336*(lr**7*rBar*sigma**11) + \
	592061462347776*(lr**6*rBar**2*sigma**11) + 52675426426880*(lr**5*rBar**3*sigma**11) + 2600353378304*(lr**4*rBar**4*sigma**11) + \
	58924875776*(lr**3*rBar**5*sigma**11) + 8365185439367168*(lr**8*sigma**12) + 2791496143405056*(lr**7*rBar*sigma**12) + \
	376741246992384*(lr**6*rBar**2*sigma**12) + 24818836553728*(lr**5*rBar**3*sigma**12) + 690024280064*(lr**4*rBar**4*sigma**12) + \
	5478775638720512*(lr**8*sigma**13) + 1482657725808640*(lr**7*rBar*sigma**13) + 145344193069056*(lr**6*rBar**2*sigma**13) + \
	5264997007360*(lr**5*rBar**3*sigma**13) + 2499420180447232*(lr**8*sigma**14) + 484740011130880*(lr**7*rBar*sigma**14) + \
	25708988612608*(lr**6*rBar**2*sigma**14) + 709609602940928*(lr**8*sigma**15) + 73600501841920*(lr**7*rBar*sigma**15) + \
	94455965155328*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : np.sqrt(2147483648*(lr)**8 + 73014444032*(lr**8*sigma) + \
	7918845952*(lr**7*rBar*sigma) + 1163885805568*(lr**8*sigma**2) + 234209935360*(lr**7*rBar*sigma**2) + \
	12813598720*(lr**6*rBar**2*sigma**2) + 11545962610688*(lr**8*sigma**3) + 3218914410496*(lr**7*rBar*sigma**3) + \
	323380838400*(lr**6*rBar**2*sigma**3) + 11899240448*(lr**5*rBar**3*sigma**3) + 79779668688896*(lr**8*sigma**4) + \
	27245798227968*(lr**7*rBar*sigma**4) + 3746448801792*(lr**6*rBar**2*sigma**4) + 249603031040*(lr**5*rBar**3*sigma**4) + \
	6946816000*(lr**4*rBar**4*sigma**4) + 407138872590336*(lr**8*sigma**5) + 158659459940352*(lr**7*rBar*sigma**5) + \
	26342247628800*(lr**6*rBar**2*sigma**5) + 2361552535552*(lr**5*rBar**3*sigma**5) + 116559708160*(lr**4*rBar**4*sigma**5) + \
	2615148544*(lr**3*rBar**5*sigma**5) + 1587375228059648*(lr**8*sigma**6) + 672355780657152*(lr**7*rBar*sigma**6) + \
	125177789939712*(lr**6*rBar**2*sigma**6) + 13267386761216*(lr**5*rBar**3*sigma**6) + 858268893184*(lr**4*rBar**4*sigma**6) + \
	33004978176*(lr**3*rBar**5*sigma**6) + 621019136*(lr**2*rBar**6*sigma**6) + 4823154074058752*(lr**8*sigma**7) + \
	2138114019819520*(lr**7*rBar*sigma**7) + 423450571702272*(lr**6*rBar**2*sigma**7) + 48999738638336*(lr**5*rBar**3*sigma**7) + \
	3620632264704*(lr**4*rBar**4*sigma**7) + 174203338752*(lr**3*rBar**5*sigma**7) + 5258608640*(lr**2*rBar**6*sigma**7) + \
	85196800*(lr*rBar**7*sigma**7) + 11542131442253824*(lr**8*sigma**8) + 5182872106565632*(lr**7*rBar*sigma**8) + \
	1045434924531712*(lr**6*rBar**2*sigma**8) + 124268989120512*(lr**5*rBar**3*sigma**8) + 9565774675968*(lr**4*rBar**4*sigma**8) + \
	491772051456*(lr**3*rBar**5*sigma**8) + 16759914496*(lr**2*rBar**6*sigma**8) + 364380160*(lr*rBar**7*sigma**8) + \
	5177344*(rBar**8*sigma**8) + 21826730911072256*(lr**8*sigma**9) + 9622892496289792*(lr**7*rBar*sigma**9) + \
	1897681168367616*(lr**6*rBar**2*sigma**9) + 219108245635072*(lr**5*rBar**3*sigma**9) + 16199545257984*(lr**4*rBar**4*sigma**9) + \
	782405861376*(lr**3*rBar**5*sigma**9) + 23789502464*(lr**2*rBar**6*sigma**9) + 389545984*(lr*rBar**7*sigma**9) + \
	32508435504300032*(lr**8*sigma**10) + 13617476213932032*(lr**7*rBar*sigma**10) + 2513300747255808*(lr**6*rBar**2*sigma**10) + \
	265142670917632*(lr**5*rBar**3*sigma**10) + 17165166510080*(lr**4*rBar**4*sigma**10) + 664695668736*(lr**3*rBar**5*sigma**10) + \
	12690784256*(lr**2*rBar**6*sigma**10) + 37732935096336384*(lr**8*sigma**11) + 14457248330809344*(lr**7*rBar*sigma**11) + \
	2368245849391104*(lr**6*rBar**2*sigma**11) + 210701705707520*(lr**5*rBar**3*sigma**11) + 10401413513216*(lr**4*rBar**4*sigma**11) + \
	235699503104*(lr**3*rBar**5*sigma**11) + 33460741757468672*(lr**8*sigma**12) + 11165984573620224*(lr**7*rBar*sigma**12) + \
	1506964987969536*(lr**6*rBar**2*sigma**12) + 99275346214912*(lr**5*rBar**3*sigma**12) + 2760097120256*(lr**4*rBar**4*sigma**12) + \
	21915102554882048*(lr**8*sigma**13) + 5930630903234560*(lr**7*rBar*sigma**13) + 581376772276224*(lr**6*rBar**2*sigma**13) + \
	21059988029440*(lr**5*rBar**3*sigma**13) + 9997680721788928*(lr**8*sigma**14) + 1938960044523520*(lr**7*rBar*sigma**14) + \
	102835954450432*(lr**6*rBar**2*sigma**14) + 2838438411763712*(lr**8*sigma**15) + 294402007367680*(lr**7*rBar*sigma**15) + \
	377823860621312*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : np.sqrt(8589934592*(lr)**8 + 292057776128*(lr**8*sigma) + \
	31675383808*(lr**7*rBar*sigma) + 4655543222272*(lr**8*sigma**2) + 936839741440*(lr**7*rBar*sigma**2) + \
	51254394880*(lr**6*rBar**2*sigma**2) + 46183850442752*(lr**8*sigma**3) + 12875657641984*(lr**7*rBar*sigma**3) + \
	1293523353600*(lr**6*rBar**2*sigma**3) + 47596961792*(lr**5*rBar**3*sigma**3) + 319118674755584*(lr**8*sigma**4) + \
	108983192911872*(lr**7*rBar*sigma**4) + 14985795207168*(lr**6*rBar**2*sigma**4) + 998412124160*(lr**5*rBar**3*sigma**4) + \
	27787264000*(lr**4*rBar**4*sigma**4) + 1628555490361344*(lr**8*sigma**5) + 634637839761408*(lr**7*rBar*sigma**5) + \
	105368990515200*(lr**6*rBar**2*sigma**5) + 9446210142208*(lr**5*rBar**3*sigma**5) + 466238832640*(lr**4*rBar**4*sigma**5) + \
	10460594176*(lr**3*rBar**5*sigma**5) + 6349500912238592*(lr**8*sigma**6) + 2689423122628608*(lr**7*rBar*sigma**6) + \
	500711159758848*(lr**6*rBar**2*sigma**6) + 53069547044864*(lr**5*rBar**3*sigma**6) + 3433075572736*(lr**4*rBar**4*sigma**6) + \
	132019912704*(lr**3*rBar**5*sigma**6) + 2484076544*(lr**2*rBar**6*sigma**6) + 19292616296235008*(lr**8*sigma**7) + \
	8552456079278080*(lr**7*rBar*sigma**7) + 1693802286809088*(lr**6*rBar**2*sigma**7) + 195998954553344*(lr**5*rBar**3*sigma**7) + \
	14482529058816*(lr**4*rBar**4*sigma**7) + 696813355008*(lr**3*rBar**5*sigma**7) + 21034434560*(lr**2*rBar**6*sigma**7) + \
	340787200*(lr*rBar**7*sigma**7) + 46168525769015296*(lr**8*sigma**8) + 20731488426262528*(lr**7*rBar*sigma**8) + \
	4181739698126848*(lr**6*rBar**2*sigma**8) + 497075956482048*(lr**5*rBar**3*sigma**8) + 38263098703872*(lr**4*rBar**4*sigma**8) + \
	1967088205824*(lr**3*rBar**5*sigma**8) + 67039657984*(lr**2*rBar**6*sigma**8) + 1457520640*(lr*rBar**7*sigma**8) + \
	20709376*(rBar**8*sigma**8) + 87306923644289024*(lr**8*sigma**9) + 38491569985159168*(lr**7*rBar*sigma**9) + \
	7590724673470464*(lr**6*rBar**2*sigma**9) + 876432982540288*(lr**5*rBar**3*sigma**9) + 64798181031936*(lr**4*rBar**4*sigma**9) + \
	3129623445504*(lr**3*rBar**5*sigma**9) + 95158009856*(lr**2*rBar**6*sigma**9) + 1558183936*(lr*rBar**7*sigma**9) + \
	130033742017200128*(lr**8*sigma**10) + 54469904855728128*(lr**7*rBar*sigma**10) + 10053202989023232*(lr**6*rBar**2*sigma**10) + \
	1060570683670528*(lr**5*rBar**3*sigma**10) + 68660666040320*(lr**4*rBar**4*sigma**10) + 2658782674944*(lr**3*rBar**5*sigma**10) + \
	50763137024*(lr**2*rBar**6*sigma**10) + 150931740385345536*(lr**8*sigma**11) + 57828993323237376*(lr**7*rBar*sigma**11) + \
	9472983397564416*(lr**6*rBar**2*sigma**11) + 842806822830080*(lr**5*rBar**3*sigma**11) + 41605654052864*(lr**4*rBar**4*sigma**11) + \
	942798012416*(lr**3*rBar**5*sigma**11) + 133842967029874688*(lr**8*sigma**12) + 44663938294480896*(lr**7*rBar*sigma**12) + \
	6027859951878144*(lr**6*rBar**2*sigma**12) + 397101384859648*(lr**5*rBar**3*sigma**12) + 11040388481024*(lr**4*rBar**4*sigma**12) + \
	87660410219528192*(lr**8*sigma**13) + 23722523612938240*(lr**7*rBar*sigma**13) + 2325507089104896*(lr**6*rBar**2*sigma**13) + \
	84239952117760*(lr**5*rBar**3*sigma**13) + 39990722887155712*(lr**8*sigma**14) + 7755840178094080*(lr**7*rBar*sigma**14) + \
	411343817801728*(lr**6*rBar**2*sigma**14) + 11353753647054848*(lr**8*sigma**15) + 1177608029470720*(lr**7*rBar*sigma**15) + \
	1511295442485248*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : np.sqrt(13107200*(lr)**10 + 552468480*(lr**10*sigma) + \
	60948480*(lr**9*rBar*sigma) + 11060854784*(lr**10*sigma**2) + 2304606208*(lr**9*rBar*sigma**2) + 127811584*(lr**8*rBar**2*sigma**2) + \
	139856543744*(lr**10*sigma**3) + 41161981952*(lr**9*rBar*sigma**3) + 4283990016*(lr**8*rBar**2*sigma**3) + \
	159318016*(lr**7*rBar**3*sigma**3) + 1252551548928*(lr**10*sigma**4) + 461395722240*(lr**9*rBar*sigma**4) + \
	67346980864*(lr**8*rBar**2*sigma**4) + 4662640640*(lr**7*rBar**3*sigma**4) + 130859008*(lr**6*rBar**4*sigma**4) + \
	8445915594752*(lr**10*sigma**5) + 3637132746752*(lr**9*rBar*sigma**5) + 659082067968*(lr**8*rBar**2*sigma**5) + \
	63411503104*(lr**7*rBar**3*sigma**5) + 3278528512*(lr**6*rBar**4*sigma**5) + 74088448*(lr**5*rBar**5*sigma**5) + \
	44489940874240*(lr**10*sigma**6) + 21410417938432*(lr**9*rBar*sigma**6) + 4493635569664*(lr**8*rBar**2*sigma**6) + \
	531077516288*(lr**7*rBar**3*sigma**6) + 37692502016*(lr**6*rBar**4*sigma**6) + 1546567680*(lr**5*rBar**5*sigma**6) + \
	29317120*(lr**4*rBar**6*sigma**6) + 187472661061632*(lr**10*sigma**7) + 97534277018112*(lr**9*rBar*sigma**7) + \
	22630701626880*(lr**8*rBar**2*sigma**7) + 3059492891136*(lr**7*rBar**3*sigma**7) + 262870009344*(lr**6*rBar**4*sigma**7) + \
	14549604352*(lr**5*rBar**5*sigma**7) + 490133504*(lr**4*rBar**6*sigma**7) + 8015872*(lr**3*rBar**7*sigma**7) + \
	641809296781312*(lr**10*sigma**8) + 351553748954368*(lr**9*rBar*sigma**8) + 87075518454016*(lr**8*rBar**2*sigma**8) + \
	12822939437312*(lr**7*rBar**3*sigma**8) + 1238228483840*(lr**6*rBar**4*sigma**8) + 81197207040*(lr**5*rBar**5*sigma**8) + \
	3591293440*(lr**4*rBar**6*sigma**8) + 100768768*(lr**3*rBar**7*sigma**8) + 1451008*(lr**2*rBar**8*sigma**8) + \
	1802710485499904*(lr**10*sigma**9) + 1016335073965184*(lr**9*rBar*sigma**9) + 261081981717120*(lr**8*rBar**2*sigma**9) + \
	40314048024320*(lr**7*rBar**3*sigma**9) + 4148988619264*(lr**6*rBar**4*sigma**9) + 297550556416*(lr**5*rBar**5*sigma**9) + \
	15052031744*(lr**4*rBar**6*sigma**9) + 528831488*(lr**3*rBar**7*sigma**9) + 12211200*(lr**2*rBar**8*sigma**9) + \
	157184*(lr*rBar**9*sigma**9) + 4177003931713536*(lr**10*sigma**10) + 2374154099398464*(lr**9*rBar*sigma**10) + \
	616433830744768*(lr**8*rBar**2*sigma**10) + 96564342310592*(lr**7*rBar**3*sigma**10) + 10137465087040*(lr**6*rBar**4*sigma**10) + \
	747814819584*(lr**5*rBar**5*sigma**10) + 39443804928*(lr**4*rBar**6*sigma**10) + 1481171904*(lr**3*rBar**7*sigma**10) + \
	38603200*(lr**2*rBar**8*sigma**10) + 665216*(lr*rBar**9*sigma**10) + 7744*(rBar**10*sigma**10) + 7998015712133120*(lr**10*sigma**11) + \
	4491791460800576*(lr**9*rBar*sigma**11) + 1149812252009952*(lr**8*rBar**2*sigma**11) + 177055060370240*(lr**7*rBar**3*sigma**11) + \
	18193628501600*(lr**6*rBar**4*sigma**11) + 1304819430560*(lr**5*rBar**5*sigma**11) + 66134731040*(lr**4*rBar**6*sigma**11) + \
	2333136032*(lr**3*rBar**7*sigma**11) + 54207840*(lr**2*rBar**8*sigma**11) + 703616*(lr*rBar**9*sigma**11) + \
	12633297946869760*(lr**10*sigma**12) + 6866279436029680*(lr**9*rBar*sigma**12) + 1688550640955104*(lr**8*rBar**2*sigma**12) + \
	247254346524112*(lr**7*rBar**3*sigma**12) + 23796727700512*(lr**6*rBar**4*sigma**12) + 1560162485424*(lr**5*rBar**5*sigma**12) + \
	69254016768*(lr**4*rBar**6*sigma**12) + 1958708464*(lr**3*rBar**7*sigma**12) + 28579744*(lr**2*rBar**8*sigma**12) + \
	16371917814497280*(lr**10*sigma**13) + 8417118908178720*(lr**9*rBar*sigma**13) + 1931617391200728*(lr**8*rBar**2*sigma**13) + \
	258832294893024*(lr**7*rBar**3*sigma**13) + 22117391959680*(lr**6*rBar**4*sigma**13) + 1223080239536*(lr**5*rBar**5*sigma**13) + \
	41404317232*(lr**4*rBar**6*sigma**13) + 685592528*(lr**3*rBar**7*sigma**13) + 17237288916090880*(lr**10*sigma**14) + \
	8162121509018752*(lr**9*rBar*sigma**14) + 1687266472697428*(lr**8*rBar**2*sigma**14) + 196927331002492*(lr**7*rBar**3*sigma**14) + \
	13862588278660*(lr**6*rBar**4*sigma**14) + 567590962584*(lr**5*rBar**5*sigma**14) + 10828300176*(lr**4*rBar**6*sigma**14) + \
	14517515511136256*(lr**10*sigma**15) + 6123095929776512*(lr**9*rBar*sigma**15) + 1087848889143456*(lr**8*rBar**2*sigma**15) + \
	102927326148774*(lr**7*rBar**3*sigma**15) + 5260572051347*(lr**6*rBar**4*sigma**15) + 118461237176*(lr**5*rBar**5*sigma**15) + \
	9551483144503296*(lr**10*sigma**16) + 3428862774592176*(lr**9*rBar*sigma**16) + 488201527696624*(lr**8*rBar**2*sigma**16) + \
	33078259384628*(lr**7*rBar**3*sigma**16) + 914281471135*(lr**6*rBar**4*sigma**16) + 4731264290521088*(lr**10*sigma**17) + \
	1349701403248664*(lr**9*rBar*sigma**17) + 136248519567816*(lr**8*rBar**2*sigma**17) + 4932215760720*(lr**7*rBar**3*sigma**17) + \
	1659932008546304*(lr**10*sigma**18) + 333255476675968*(lr**9*rBar*sigma**18) + 17814927959752*(lr**8*rBar**2*sigma**18) + \
	367792399699968*(lr**10*sigma**19) + 38845548716112*(lr**9*rBar*sigma**19) + 38706485829632*(lr**10*sigma**20)),\
	lambda \
	sigma,lr,rBar : np.sqrt(52428800*(lr)**10 + 2209873920*(lr**10*sigma) + 243793920*(lr**9*rBar*sigma) + 44243419136*(lr**10*sigma**2) + \
	9218424832*(lr**9*rBar*sigma**2) + 511246336*(lr**8*rBar**2*sigma**2) + 559426174976*(lr**10*sigma**3) + \
	164647927808*(lr**9*rBar*sigma**3) + 17135960064*(lr**8*rBar**2*sigma**3) + 637272064*(lr**7*rBar**3*sigma**3) + \
	5010206195712*(lr**10*sigma**4) + 1845582888960*(lr**9*rBar*sigma**4) + 269387923456*(lr**8*rBar**2*sigma**4) + \
	18650562560*(lr**7*rBar**3*sigma**4) + 523436032*(lr**6*rBar**4*sigma**4) + 33783662379008*(lr**10*sigma**5) + \
	14548530987008*(lr**9*rBar*sigma**5) + 2636328271872*(lr**8*rBar**2*sigma**5) + 253646012416*(lr**7*rBar**3*sigma**5) + \
	13114114048*(lr**6*rBar**4*sigma**5) + 296353792*(lr**5*rBar**5*sigma**5) + 177959763496960*(lr**10*sigma**6) + \
	85641671753728*(lr**9*rBar*sigma**6) + 17974542278656*(lr**8*rBar**2*sigma**6) + 2124310065152*(lr**7*rBar**3*sigma**6) + \
	150770008064*(lr**6*rBar**4*sigma**6) + 6186270720*(lr**5*rBar**5*sigma**6) + 117268480*(lr**4*rBar**6*sigma**6) + \
	749890644246528*(lr**10*sigma**7) + 390137108072448*(lr**9*rBar*sigma**7) + 90522806507520*(lr**8*rBar**2*sigma**7) + \
	12237971564544*(lr**7*rBar**3*sigma**7) + 1051480037376*(lr**6*rBar**4*sigma**7) + 58198417408*(lr**5*rBar**5*sigma**7) + \
	1960534016*(lr**4*rBar**6*sigma**7) + 32063488*(lr**3*rBar**7*sigma**7) + 2567237187125248*(lr**10*sigma**8) + \
	1406214995817472*(lr**9*rBar*sigma**8) + 348302073816064*(lr**8*rBar**2*sigma**8) + 51291757749248*(lr**7*rBar**3*sigma**8) + \
	4952913935360*(lr**6*rBar**4*sigma**8) + 324788828160*(lr**5*rBar**5*sigma**8) + 14365173760*(lr**4*rBar**6*sigma**8) + \
	403075072*(lr**3*rBar**7*sigma**8) + 5804032*(lr**2*rBar**8*sigma**8) + 7210841941999616*(lr**10*sigma**9) + \
	4065340295860736*(lr**9*rBar*sigma**9) + 1044327926868480*(lr**8*rBar**2*sigma**9) + 161256192097280*(lr**7*rBar**3*sigma**9) + \
	16595954477056*(lr**6*rBar**4*sigma**9) + 1190202225664*(lr**5*rBar**5*sigma**9) + 60208126976*(lr**4*rBar**6*sigma**9) + \
	2115325952*(lr**3*rBar**7*sigma**9) + 48844800*(lr**2*rBar**8*sigma**9) + 628736*(lr*rBar**9*sigma**9) + \
	16708015726854144*(lr**10*sigma**10) + 9496616397593856*(lr**9*rBar*sigma**10) + 2465735322979072*(lr**8*rBar**2*sigma**10) + \
	386257369242368*(lr**7*rBar**3*sigma**10) + 40549860348160*(lr**6*rBar**4*sigma**10) + 2991259278336*(lr**5*rBar**5*sigma**10) + \
	157775219712*(lr**4*rBar**6*sigma**10) + 5924687616*(lr**3*rBar**7*sigma**10) + 154412800*(lr**2*rBar**8*sigma**10) + \
	2660864*(lr*rBar**9*sigma**10) + 30976*(rBar**10*sigma**10) + 31992062848532480*(lr**10*sigma**11) + \
	17967165843202304*(lr**9*rBar*sigma**11) + 4599249008039808*(lr**8*rBar**2*sigma**11) + 708220241480960*(lr**7*rBar**3*sigma**11) + \
	72774514006400*(lr**6*rBar**4*sigma**11) + 5219277722240*(lr**5*rBar**5*sigma**11) + 264538924160*(lr**4*rBar**6*sigma**11) + \
	9332544128*(lr**3*rBar**7*sigma**11) + 216831360*(lr**2*rBar**8*sigma**11) + 2814464*(lr*rBar**9*sigma**11) + \
	50533191787479040*(lr**10*sigma**12) + 27465117744118720*(lr**9*rBar*sigma**12) + 6754202563820416*(lr**8*rBar**2*sigma**12) + \
	989017386096448*(lr**7*rBar**3*sigma**12) + 95186910802048*(lr**6*rBar**4*sigma**12) + 6240649941696*(lr**5*rBar**5*sigma**12) + \
	277016067072*(lr**4*rBar**6*sigma**12) + 7834833856*(lr**3*rBar**7*sigma**12) + 114318976*(lr**2*rBar**8*sigma**12) + \
	65487671257989120*(lr**10*sigma**13) + 33668475632714880*(lr**9*rBar*sigma**13) + 7726469564802912*(lr**8*rBar**2*sigma**13) + \
	1035329179572096*(lr**7*rBar**3*sigma**13) + 88469567838720*(lr**6*rBar**4*sigma**13) + 4892320958144*(lr**5*rBar**5*sigma**13) + \
	165617268928*(lr**4*rBar**6*sigma**13) + 2742370112*(lr**3*rBar**7*sigma**13) + 68949155664363520*(lr**10*sigma**14) + \
	32648486036075008*(lr**9*rBar*sigma**14) + 6749065890789712*(lr**8*rBar**2*sigma**14) + 787709324009968*(lr**7*rBar**3*sigma**14) + \
	55450353114640*(lr**6*rBar**4*sigma**14) + 2270363850336*(lr**5*rBar**5*sigma**14) + 43313200704*(lr**4*rBar**6*sigma**14) + \
	58070062044545024*(lr**10*sigma**15) + 24492383719106048*(lr**9*rBar*sigma**15) + 4351395556573824*(lr**8*rBar**2*sigma**15) + \
	411709304595096*(lr**7*rBar**3*sigma**15) + 21042288205388*(lr**6*rBar**4*sigma**15) + 473844948704*(lr**5*rBar**5*sigma**15) + \
	38205932578013184*(lr**10*sigma**16) + 13715451098368704*(lr**9*rBar*sigma**16) + 1952806110786496*(lr**8*rBar**2*sigma**16) + \
	132313037538512*(lr**7*rBar**3*sigma**16) + 3657125884540*(lr**6*rBar**4*sigma**16) + 18925057162084352*(lr**10*sigma**17) + \
	5398805612994656*(lr**9*rBar*sigma**17) + 544994078271264*(lr**8*rBar**2*sigma**17) + 19728863042880*(lr**7*rBar**3*sigma**17) + \
	6639728034185216*(lr**10*sigma**18) + 1333021906703872*(lr**9*rBar*sigma**18) + 71259711839008*(lr**8*rBar**2*sigma**18) + \
	1471169598799872*(lr**10*sigma**19) + 155382194864448*(lr**9*rBar*sigma**19) + 154825943318528*(lr**10*sigma**20)),\
	lambda \
	sigma,lr,rBar : np.sqrt(209715200*(lr)**10 + 8839495680*(lr**10*sigma) + 975175680*(lr**9*rBar*sigma) + 176973676544*(lr**10*sigma**2) \
	+ 36873699328*(lr**9*rBar*sigma**2) + 2044985344*(lr**8*rBar**2*sigma**2) + 2237704699904*(lr**10*sigma**3) + \
	658591711232*(lr**9*rBar*sigma**3) + 68543840256*(lr**8*rBar**2*sigma**3) + 2549088256*(lr**7*rBar**3*sigma**3) + \
	20040824782848*(lr**10*sigma**4) + 7382331555840*(lr**9*rBar*sigma**4) + 1077551693824*(lr**8*rBar**2*sigma**4) + \
	74602250240*(lr**7*rBar**3*sigma**4) + 2093744128*(lr**6*rBar**4*sigma**4) + 135134649516032*(lr**10*sigma**5) + \
	58194123948032*(lr**9*rBar*sigma**5) + 10545313087488*(lr**8*rBar**2*sigma**5) + 1014584049664*(lr**7*rBar**3*sigma**5) + \
	52456456192*(lr**6*rBar**4*sigma**5) + 1185415168*(lr**5*rBar**5*sigma**5) + 711839053987840*(lr**10*sigma**6) + \
	342566687014912*(lr**9*rBar*sigma**6) + 71898169114624*(lr**8*rBar**2*sigma**6) + 8497240260608*(lr**7*rBar**3*sigma**6) + \
	603080032256*(lr**6*rBar**4*sigma**6) + 24745082880*(lr**5*rBar**5*sigma**6) + 469073920*(lr**4*rBar**6*sigma**6) + \
	2999562576986112*(lr**10*sigma**7) + 1560548432289792*(lr**9*rBar*sigma**7) + 362091226030080*(lr**8*rBar**2*sigma**7) + \
	48951886258176*(lr**7*rBar**3*sigma**7) + 4205920149504*(lr**6*rBar**4*sigma**7) + 232793669632*(lr**5*rBar**5*sigma**7) + \
	7842136064*(lr**4*rBar**6*sigma**7) + 128253952*(lr**3*rBar**7*sigma**7) + 10268948748500992*(lr**10*sigma**8) + \
	5624859983269888*(lr**9*rBar*sigma**8) + 1393208295264256*(lr**8*rBar**2*sigma**8) + 205167030996992*(lr**7*rBar**3*sigma**8) + \
	19811655741440*(lr**6*rBar**4*sigma**8) + 1299155312640*(lr**5*rBar**5*sigma**8) + 57460695040*(lr**4*rBar**6*sigma**8) + \
	1612300288*(lr**3*rBar**7*sigma**8) + 23216128*(lr**2*rBar**8*sigma**8) + 28843367767998464*(lr**10*sigma**9) + \
	16261361183442944*(lr**9*rBar*sigma**9) + 4177311707473920*(lr**8*rBar**2*sigma**9) + 645024768389120*(lr**7*rBar**3*sigma**9) + \
	66383817908224*(lr**6*rBar**4*sigma**9) + 4760808902656*(lr**5*rBar**5*sigma**9) + 240832507904*(lr**4*rBar**6*sigma**9) + \
	8461303808*(lr**3*rBar**7*sigma**9) + 195379200*(lr**2*rBar**8*sigma**9) + 2514944*(lr*rBar**9*sigma**9) + \
	66832062907416576*(lr**10*sigma**10) + 37986465590375424*(lr**9*rBar*sigma**10) + 9862941291916288*(lr**8*rBar**2*sigma**10) + \
	1545029476969472*(lr**7*rBar**3*sigma**10) + 162199441392640*(lr**6*rBar**4*sigma**10) + 11965037113344*(lr**5*rBar**5*sigma**10) + \
	631100878848*(lr**4*rBar**6*sigma**10) + 23698750464*(lr**3*rBar**7*sigma**10) + 617651200*(lr**2*rBar**8*sigma**10) + \
	10643456*(lr*rBar**9*sigma**10) + 123904*(rBar**10*sigma**10) + 127968251394129920*(lr**10*sigma**11) + \
	71868663372809216*(lr**9*rBar*sigma**11) + 18396996032159232*(lr**8*rBar**2*sigma**11) + 2832880965923840*(lr**7*rBar**3*sigma**11) + \
	291098056025600*(lr**6*rBar**4*sigma**11) + 20877110888960*(lr**5*rBar**5*sigma**11) + 1058155696640*(lr**4*rBar**6*sigma**11) + \
	37330176512*(lr**3*rBar**7*sigma**11) + 867325440*(lr**2*rBar**8*sigma**11) + 11257856*(lr*rBar**9*sigma**11) + \
	202132767149916160*(lr**10*sigma**12) + 109860470976474880*(lr**9*rBar*sigma**12) + 27016810255281664*(lr**8*rBar**2*sigma**12) + \
	3956069544385792*(lr**7*rBar**3*sigma**12) + 380747643208192*(lr**6*rBar**4*sigma**12) + 24962599766784*(lr**5*rBar**5*sigma**12) + \
	1108064268288*(lr**4*rBar**6*sigma**12) + 31339335424*(lr**3*rBar**7*sigma**12) + 457275904*(lr**2*rBar**8*sigma**12) + \
	261950685031956480*(lr**10*sigma**13) + 134673902530859520*(lr**9*rBar*sigma**13) + 30905878259211648*(lr**8*rBar**2*sigma**13) + \
	4141316718288384*(lr**7*rBar**3*sigma**13) + 353878271354880*(lr**6*rBar**4*sigma**13) + 19569283832576*(lr**5*rBar**5*sigma**13) + \
	662469075712*(lr**4*rBar**6*sigma**13) + 10969480448*(lr**3*rBar**7*sigma**13) + 275796622657454080*(lr**10*sigma**14) + \
	130593944144300032*(lr**9*rBar*sigma**14) + 26996263563158848*(lr**8*rBar**2*sigma**14) + 3150837296039872*(lr**7*rBar**3*sigma**14) + \
	221801412458560*(lr**6*rBar**4*sigma**14) + 9081455401344*(lr**5*rBar**5*sigma**14) + 173252802816*(lr**4*rBar**6*sigma**14) + \
	232280248178180096*(lr**10*sigma**15) + 97969534876424192*(lr**9*rBar*sigma**15) + 17405582226295296*(lr**8*rBar**2*sigma**15) + \
	1646837218380384*(lr**7*rBar**3*sigma**15) + 84169152821552*(lr**6*rBar**4*sigma**15) + 1895379794816*(lr**5*rBar**5*sigma**15) + \
	152823730312052736*(lr**10*sigma**16) + 54861804393474816*(lr**9*rBar*sigma**16) + 7811224443145984*(lr**8*rBar**2*sigma**16) + \
	529252150154048*(lr**7*rBar**3*sigma**16) + 14628503538160*(lr**6*rBar**4*sigma**16) + 75700228648337408*(lr**10*sigma**17) + \
	21595222451978624*(lr**9*rBar*sigma**17) + 2179976313085056*(lr**8*rBar**2*sigma**17) + 78915452171520*(lr**7*rBar**3*sigma**17) + \
	26558912136740864*(lr**10*sigma**18) + 5332087626815488*(lr**9*rBar*sigma**18) + 285038847356032*(lr**8*rBar**2*sigma**18) + \
	5884678395199488*(lr**10*sigma**19) + 621528779457792*(lr**9*rBar*sigma**19) + 619303773274112*(lr**10*sigma**20)),\
	lambda \
	sigma,lr,rBar : np.sqrt(838860800*(lr)**10 + 35357982720*(lr**10*sigma) + 3900702720*(lr**9*rBar*sigma) + \
	707894706176*(lr**10*sigma**2) + 147494797312*(lr**9*rBar*sigma**2) + 8179941376*(lr**8*rBar**2*sigma**2) + \
	8950818799616*(lr**10*sigma**3) + 2634366844928*(lr**9*rBar*sigma**3) + 274175361024*(lr**8*rBar**2*sigma**3) + \
	10196353024*(lr**7*rBar**3*sigma**3) + 80163299131392*(lr**10*sigma**4) + 29529326223360*(lr**9*rBar*sigma**4) + \
	4310206775296*(lr**8*rBar**2*sigma**4) + 298409000960*(lr**7*rBar**3*sigma**4) + 8374976512*(lr**6*rBar**4*sigma**4) + \
	540538598064128*(lr**10*sigma**5) + 232776495792128*(lr**9*rBar*sigma**5) + 42181252349952*(lr**8*rBar**2*sigma**5) + \
	4058336198656*(lr**7*rBar**3*sigma**5) + 209825824768*(lr**6*rBar**4*sigma**5) + 4741660672*(lr**5*rBar**5*sigma**5) + \
	2847356215951360*(lr**10*sigma**6) + 1370266748059648*(lr**9*rBar*sigma**6) + 287592676458496*(lr**8*rBar**2*sigma**6) + \
	33988961042432*(lr**7*rBar**3*sigma**6) + 2412320129024*(lr**6*rBar**4*sigma**6) + 98980331520*(lr**5*rBar**5*sigma**6) + \
	1876295680*(lr**4*rBar**6*sigma**6) + 11998250307944448*(lr**10*sigma**7) + 6242193729159168*(lr**9*rBar*sigma**7) + \
	1448364904120320*(lr**8*rBar**2*sigma**7) + 195807545032704*(lr**7*rBar**3*sigma**7) + 16823680598016*(lr**6*rBar**4*sigma**7) + \
	931174678528*(lr**5*rBar**5*sigma**7) + 31368544256*(lr**4*rBar**6*sigma**7) + 513015808*(lr**3*rBar**7*sigma**7) + \
	41075794994003968*(lr**10*sigma**8) + 22499439933079552*(lr**9*rBar*sigma**8) + 5572833181057024*(lr**8*rBar**2*sigma**8) + \
	820668123987968*(lr**7*rBar**3*sigma**8) + 79246622965760*(lr**6*rBar**4*sigma**8) + 5196621250560*(lr**5*rBar**5*sigma**8) + \
	229842780160*(lr**4*rBar**6*sigma**8) + 6449201152*(lr**3*rBar**7*sigma**8) + 92864512*(lr**2*rBar**8*sigma**8) + \
	115373471071993856*(lr**10*sigma**9) + 65045444733771776*(lr**9*rBar*sigma**9) + 16709246829895680*(lr**8*rBar**2*sigma**9) + \
	2580099073556480*(lr**7*rBar**3*sigma**9) + 265535271632896*(lr**6*rBar**4*sigma**9) + 19043235610624*(lr**5*rBar**5*sigma**9) + \
	963330031616*(lr**4*rBar**6*sigma**9) + 33845215232*(lr**3*rBar**7*sigma**9) + 781516800*(lr**2*rBar**8*sigma**9) + \
	10059776*(lr*rBar**9*sigma**9) + 267328251629666304*(lr**10*sigma**10) + 151945862361501696*(lr**9*rBar*sigma**10) + \
	39451765167665152*(lr**8*rBar**2*sigma**10) + 6180117907877888*(lr**7*rBar**3*sigma**10) + 648797765570560*(lr**6*rBar**4*sigma**10) + \
	47860148453376*(lr**5*rBar**5*sigma**10) + 2524403515392*(lr**4*rBar**6*sigma**10) + 94795001856*(lr**3*rBar**7*sigma**10) + \
	2470604800*(lr**2*rBar**8*sigma**10) + 42573824*(lr*rBar**9*sigma**10) + 495616*(rBar**10*sigma**10) + \
	511873005576519680*(lr**10*sigma**11) + 287474653491236864*(lr**9*rBar*sigma**11) + 73587984128636928*(lr**8*rBar**2*sigma**11) + \
	11331523863695360*(lr**7*rBar**3*sigma**11) + 1164392224102400*(lr**6*rBar**4*sigma**11) + 83508443555840*(lr**5*rBar**5*sigma**11) + \
	4232622786560*(lr**4*rBar**6*sigma**11) + 149320706048*(lr**3*rBar**7*sigma**11) + 3469301760*(lr**2*rBar**8*sigma**11) + \
	45031424*(lr*rBar**9*sigma**11) + 808531068599664640*(lr**10*sigma**12) + 439441883905899520*(lr**9*rBar*sigma**12) + \
	108067241021126656*(lr**8*rBar**2*sigma**12) + 15824278177543168*(lr**7*rBar**3*sigma**12) + \
	1522990572832768*(lr**6*rBar**4*sigma**12) + 99850399067136*(lr**5*rBar**5*sigma**12) + 4432257073152*(lr**4*rBar**6*sigma**12) + \
	125357341696*(lr**3*rBar**7*sigma**12) + 1829103616*(lr**2*rBar**8*sigma**12) + 1047802740127825920*(lr**10*sigma**13) + \
	538695610123438080*(lr**9*rBar*sigma**13) + 123623513036846592*(lr**8*rBar**2*sigma**13) + 16565266873153536*(lr**7*rBar**3*sigma**13) \
	+ 1415513085419520*(lr**6*rBar**4*sigma**13) + 78277135330304*(lr**5*rBar**5*sigma**13) + 2649876302848*(lr**4*rBar**6*sigma**13) + \
	43877921792*(lr**3*rBar**7*sigma**13) + 1103186490629816320*(lr**10*sigma**14) + 522375776577200128*(lr**9*rBar*sigma**14) + \
	107985054252635392*(lr**8*rBar**2*sigma**14) + 12603349184159488*(lr**7*rBar**3*sigma**14) + 887205649834240*(lr**6*rBar**4*sigma**14) \
	+ 36325821605376*(lr**5*rBar**5*sigma**14) + 693011211264*(lr**4*rBar**6*sigma**14) + 929120992712720384*(lr**10*sigma**15) + \
	391878139505696768*(lr**9*rBar*sigma**15) + 69622328905181184*(lr**8*rBar**2*sigma**15) + 6587348873521536*(lr**7*rBar**3*sigma**15) + \
	336676611286208*(lr**6*rBar**4*sigma**15) + 7581519179264*(lr**5*rBar**5*sigma**15) + 611294921248210944*(lr**10*sigma**16) + \
	219447217573899264*(lr**9*rBar*sigma**16) + 31244897772583936*(lr**8*rBar**2*sigma**16) + 2117008600616192*(lr**7*rBar**3*sigma**16) + \
	58514014152640*(lr**6*rBar**4*sigma**16) + 302800914593349632*(lr**10*sigma**17) + 86380889807914496*(lr**9*rBar*sigma**17) + \
	8719905252340224*(lr**8*rBar**2*sigma**17) + 315661808686080*(lr**7*rBar**3*sigma**17) + 106235648546963456*(lr**10*sigma**18) + \
	21328350507261952*(lr**9*rBar*sigma**18) + 1140155389424128*(lr**8*rBar**2*sigma**18) + 23538713580797952*(lr**10*sigma**19) + \
	2486115117831168*(lr**9*rBar*sigma**19) + 2477215093096448*(lr**10*sigma**20)),\
	lambda sigma,lr,rBar : np.sqrt(3355443200*(lr)**10 + \
	141431930880*(lr**10*sigma) + 15602810880*(lr**9*rBar*sigma) + 2831578824704*(lr**10*sigma**2) + 589979189248*(lr**9*rBar*sigma**2) + \
	32719765504*(lr**8*rBar**2*sigma**2) + 35803275198464*(lr**10*sigma**3) + 10537467379712*(lr**9*rBar*sigma**3) + \
	1096701444096*(lr**8*rBar**2*sigma**3) + 40785412096*(lr**7*rBar**3*sigma**3) + 320653196525568*(lr**10*sigma**4) + \
	118117304893440*(lr**9*rBar*sigma**4) + 17240827101184*(lr**8*rBar**2*sigma**4) + 1193636003840*(lr**7*rBar**3*sigma**4) + \
	33499906048*(lr**6*rBar**4*sigma**4) + 2162154392256512*(lr**10*sigma**5) + 931105983168512*(lr**9*rBar*sigma**5) + \
	168725009399808*(lr**8*rBar**2*sigma**5) + 16233344794624*(lr**7*rBar**3*sigma**5) + 839303299072*(lr**6*rBar**4*sigma**5) + \
	18966642688*(lr**5*rBar**5*sigma**5) + 11389424863805440*(lr**10*sigma**6) + 5481066992238592*(lr**9*rBar*sigma**6) + \
	1150370705833984*(lr**8*rBar**2*sigma**6) + 135955844169728*(lr**7*rBar**3*sigma**6) + 9649280516096*(lr**6*rBar**4*sigma**6) + \
	395921326080*(lr**5*rBar**5*sigma**6) + 7505182720*(lr**4*rBar**6*sigma**6) + 47993001231777792*(lr**10*sigma**7) + \
	24968774916636672*(lr**9*rBar*sigma**7) + 5793459616481280*(lr**8*rBar**2*sigma**7) + 783230180130816*(lr**7*rBar**3*sigma**7) + \
	67294722392064*(lr**6*rBar**4*sigma**7) + 3724698714112*(lr**5*rBar**5*sigma**7) + 125474177024*(lr**4*rBar**6*sigma**7) + \
	2052063232*(lr**3*rBar**7*sigma**7) + 164303179976015872*(lr**10*sigma**8) + 89997759732318208*(lr**9*rBar*sigma**8) + \
	22291332724228096*(lr**8*rBar**2*sigma**8) + 3282672495951872*(lr**7*rBar**3*sigma**8) + 316986491863040*(lr**6*rBar**4*sigma**8) + \
	20786485002240*(lr**5*rBar**5*sigma**8) + 919371120640*(lr**4*rBar**6*sigma**8) + 25796804608*(lr**3*rBar**7*sigma**8) + \
	371458048*(lr**2*rBar**8*sigma**8) + 461493884287975424*(lr**10*sigma**9) + 260181778935087104*(lr**9*rBar*sigma**9) + \
	66836987319582720*(lr**8*rBar**2*sigma**9) + 10320396294225920*(lr**7*rBar**3*sigma**9) + 1062141086531584*(lr**6*rBar**4*sigma**9) + \
	76172942442496*(lr**5*rBar**5*sigma**9) + 3853320126464*(lr**4*rBar**6*sigma**9) + 135380860928*(lr**3*rBar**7*sigma**9) + \
	3126067200*(lr**2*rBar**8*sigma**9) + 40239104*(lr*rBar**9*sigma**9) + 1069313006518665216*(lr**10*sigma**10) + \
	607783449446006784*(lr**9*rBar*sigma**10) + 157807060670660608*(lr**8*rBar**2*sigma**10) + 24720471631511552*(lr**7*rBar**3*sigma**10) \
	+ 2595191062282240*(lr**6*rBar**4*sigma**10) + 191440593813504*(lr**5*rBar**5*sigma**10) + 10097614061568*(lr**4*rBar**6*sigma**10) + \
	379180007424*(lr**3*rBar**7*sigma**10) + 9882419200*(lr**2*rBar**8*sigma**10) + 170295296*(lr*rBar**9*sigma**10) + \
	1982464*(rBar**10*sigma**10) + 2047492022306078720*(lr**10*sigma**11) + 1149898613964947456*(lr**9*rBar*sigma**11) + \
	294351936514547712*(lr**8*rBar**2*sigma**11) + 45326095454781440*(lr**7*rBar**3*sigma**11) + \
	4657568896409600*(lr**6*rBar**4*sigma**11) + 334033774223360*(lr**5*rBar**5*sigma**11) + 16930491146240*(lr**4*rBar**6*sigma**11) + \
	597282824192*(lr**3*rBar**7*sigma**11) + 13877207040*(lr**2*rBar**8*sigma**11) + 180125696*(lr*rBar**9*sigma**11) + \
	3234124274398658560*(lr**10*sigma**12) + 1757767535623598080*(lr**9*rBar*sigma**12) + 432268964084506624*(lr**8*rBar**2*sigma**12) + \
	63297112710172672*(lr**7*rBar**3*sigma**12) + 6091962291331072*(lr**6*rBar**4*sigma**12) + 399401596268544*(lr**5*rBar**5*sigma**12) + \
	17729028292608*(lr**4*rBar**6*sigma**12) + 501429366784*(lr**3*rBar**7*sigma**12) + 7316414464*(lr**2*rBar**8*sigma**12) + \
	4191210960511303680*(lr**10*sigma**13) + 2154782440493752320*(lr**9*rBar*sigma**13) + 494494052147386368*(lr**8*rBar**2*sigma**13) + \
	66261067492614144*(lr**7*rBar**3*sigma**13) + 5662052341678080*(lr**6*rBar**4*sigma**13) + 313108541321216*(lr**5*rBar**5*sigma**13) + \
	10599505211392*(lr**4*rBar**6*sigma**13) + 175511687168*(lr**3*rBar**7*sigma**13) + 4412745962519265280*(lr**10*sigma**14) + \
	2089503106308800512*(lr**9*rBar*sigma**14) + 431940217010541568*(lr**8*rBar**2*sigma**14) + \
	50413396736637952*(lr**7*rBar**3*sigma**14) + 3548822599336960*(lr**6*rBar**4*sigma**14) + 145303286421504*(lr**5*rBar**5*sigma**14) + \
	2772044845056*(lr**4*rBar**6*sigma**14) + 3716483970850881536*(lr**10*sigma**15) + 1567512558022787072*(lr**9*rBar*sigma**15) + \
	278489315620724736*(lr**8*rBar**2*sigma**15) + 26349395494086144*(lr**7*rBar**3*sigma**15) + \
	1346706445144832*(lr**6*rBar**4*sigma**15) + 30326076717056*(lr**5*rBar**5*sigma**15) + 2445179684992843776*(lr**10*sigma**16) + \
	877788870295597056*(lr**9*rBar*sigma**16) + 124979591090335744*(lr**8*rBar**2*sigma**16) + 8468034402464768*(lr**7*rBar**3*sigma**16) \
	+ 234056056610560*(lr**6*rBar**4*sigma**16) + 1211203658373398528*(lr**10*sigma**17) + 345523559231657984*(lr**9*rBar*sigma**17) + \
	34879621009360896*(lr**8*rBar**2*sigma**17) + 1262647234744320*(lr**7*rBar**3*sigma**17) + 424942594187853824*(lr**10*sigma**18) + \
	85313402029047808*(lr**9*rBar*sigma**18) + 4560621557696512*(lr**8*rBar**2*sigma**18) + 94154854323191808*(lr**10*sigma**19) + \
	9944460471324672*(lr**9*rBar*sigma**19) + 9908860372385792*(lr**10*sigma**20)),\
	lambda sigma,lr,rBar : np.sqrt(13421772800*(lr)**10 \
	+ 565727723520*(lr**10*sigma) + 62411243520*(lr**9*rBar*sigma) + 11326315298816*(lr**10*sigma**2) + \
	2359916756992*(lr**9*rBar*sigma**2) + 130879062016*(lr**8*rBar**2*sigma**2) + 143213100793856*(lr**10*sigma**3) + \
	42149869518848*(lr**9*rBar*sigma**3) + 4386805776384*(lr**8*rBar**2*sigma**3) + 163141648384*(lr**7*rBar**3*sigma**3) + \
	1282612786102272*(lr**10*sigma**4) + 472469219573760*(lr**9*rBar*sigma**4) + 68963308404736*(lr**8*rBar**2*sigma**4) + \
	4774544015360*(lr**7*rBar**3*sigma**4) + 133999624192*(lr**6*rBar**4*sigma**4) + 8648617569026048*(lr**10*sigma**5) + \
	3724423932674048*(lr**9*rBar*sigma**5) + 674900037599232*(lr**8*rBar**2*sigma**5) + 64933379178496*(lr**7*rBar**3*sigma**5) + \
	3357213196288*(lr**6*rBar**4*sigma**5) + 75866570752*(lr**5*rBar**5*sigma**5) + 45557699455221760*(lr**10*sigma**6) + \
	21924267968954368*(lr**9*rBar*sigma**6) + 4601482823335936*(lr**8*rBar**2*sigma**6) + 543823376678912*(lr**7*rBar**3*sigma**6) + \
	38597122064384*(lr**6*rBar**4*sigma**6) + 1583685304320*(lr**5*rBar**5*sigma**6) + 30020730880*(lr**4*rBar**6*sigma**6) + \
	191972004927111168*(lr**10*sigma**7) + 99875099666546688*(lr**9*rBar*sigma**7) + 23173838465925120*(lr**8*rBar**2*sigma**7) + \
	3132920720523264*(lr**7*rBar**3*sigma**7) + 269178889568256*(lr**6*rBar**4*sigma**7) + 14898794856448*(lr**5*rBar**5*sigma**7) + \
	501896708096*(lr**4*rBar**6*sigma**7) + 8208252928*(lr**3*rBar**7*sigma**7) + 657212719904063488*(lr**10*sigma**8) + \
	359991038929272832*(lr**9*rBar*sigma**8) + 89165330896912384*(lr**8*rBar**2*sigma**8) + 13130689983807488*(lr**7*rBar**3*sigma**8) + \
	1267945967452160*(lr**6*rBar**4*sigma**8) + 83145940008960*(lr**5*rBar**5*sigma**8) + 3677484482560*(lr**4*rBar**6*sigma**8) + \
	103187218432*(lr**3*rBar**7*sigma**8) + 1485832192*(lr**2*rBar**8*sigma**8) + 1845975537151901696*(lr**10*sigma**9) + \
	1040727115740348416*(lr**9*rBar*sigma**9) + 267347949278330880*(lr**8*rBar**2*sigma**9) + 41281585176903680*(lr**7*rBar**3*sigma**9) + \
	4248564346126336*(lr**6*rBar**4*sigma**9) + 304691769769984*(lr**5*rBar**5*sigma**9) + 15413280505856*(lr**4*rBar**6*sigma**9) + \
	541523443712*(lr**3*rBar**7*sigma**9) + 12504268800*(lr**2*rBar**8*sigma**9) + 160956416*(lr*rBar**9*sigma**9) + \
	4277252026074660864*(lr**10*sigma**10) + 2431133797784027136*(lr**9*rBar*sigma**10) + 631228242682642432*(lr**8*rBar**2*sigma**10) + \
	98881886526046208*(lr**7*rBar**3*sigma**10) + 10380764249128960*(lr**6*rBar**4*sigma**10) + 765762375254016*(lr**5*rBar**5*sigma**10) \
	+ 40390456246272*(lr**4*rBar**6*sigma**10) + 1516720029696*(lr**3*rBar**7*sigma**10) + 39529676800*(lr**2*rBar**8*sigma**10) + \
	681181184*(lr*rBar**9*sigma**10) + 7929856*(rBar**10*sigma**10) + 8189968089224314880*(lr**10*sigma**11) + \
	4599594455859789824*(lr**9*rBar*sigma**11) + 1177407746058190848*(lr**8*rBar**2*sigma**11) + \
	181304381819125760*(lr**7*rBar**3*sigma**11) + 18630275585638400*(lr**6*rBar**4*sigma**11) + \
	1336135096893440*(lr**5*rBar**5*sigma**11) + 67721964584960*(lr**4*rBar**6*sigma**11) + 2389131296768*(lr**3*rBar**7*sigma**11) + \
	55508828160*(lr**2*rBar**8*sigma**11) + 720502784*(lr*rBar**9*sigma**11) + 12936497097594634240*(lr**10*sigma**12) + \
	7031070142494392320*(lr**9*rBar*sigma**12) + 1729075856338026496*(lr**8*rBar**2*sigma**12) + \
	253188450840690688*(lr**7*rBar**3*sigma**12) + 24367849165324288*(lr**6*rBar**4*sigma**12) + \
	1597606385074176*(lr**5*rBar**5*sigma**12) + 70916113170432*(lr**4*rBar**6*sigma**12) + 2005717467136*(lr**3*rBar**7*sigma**12) + \
	29265657856*(lr**2*rBar**8*sigma**12) + 16764843842045214720*(lr**10*sigma**13) + 8619129761975009280*(lr**9*rBar*sigma**13) + \
	1977976208589545472*(lr**8*rBar**2*sigma**13) + 265044269970456576*(lr**7*rBar**3*sigma**13) + \
	22648209366712320*(lr**6*rBar**4*sigma**13) + 1252434165284864*(lr**5*rBar**5*sigma**13) + 42398020845568*(lr**4*rBar**6*sigma**13) + \
	702046748672*(lr**3*rBar**7*sigma**13) + 17650983850077061120*(lr**10*sigma**14) + 8358012425235202048*(lr**9*rBar*sigma**14) + \
	1727760868042166272*(lr**8*rBar**2*sigma**14) + 201653586946551808*(lr**7*rBar**3*sigma**14) + \
	14195290397347840*(lr**6*rBar**4*sigma**14) + 581213145686016*(lr**5*rBar**5*sigma**14) + 11088179380224*(lr**4*rBar**6*sigma**14) + \
	14865935883403526144*(lr**10*sigma**15) + 6270050232091148288*(lr**9*rBar*sigma**15) + 1113957262482898944*(lr**8*rBar**2*sigma**15) + \
	105397581976344576*(lr**7*rBar**3*sigma**15) + 5386825780579328*(lr**6*rBar**4*sigma**15) + 121304306868224*(lr**5*rBar**5*sigma**15) \
	+ 9780718739971375104*(lr**10*sigma**16) + 3511155481182388224*(lr**9*rBar*sigma**16) + 499918364361342976*(lr**8*rBar**2*sigma**16) + \
	33872137609859072*(lr**7*rBar**3*sigma**16) + 936224226442240*(lr**6*rBar**4*sigma**16) + 4844814633493594112*(lr**10*sigma**17) + \
	1382094236926631936*(lr**9*rBar*sigma**17) + 139518484037443584*(lr**8*rBar**2*sigma**17) + 5050588938977280*(lr**7*rBar**3*sigma**17) \
	+ 1699770376751415296*(lr**10*sigma**18) + 341253608116191232*(lr**9*rBar*sigma**18) + 18242486230786048*(lr**8*rBar**2*sigma**18) + \
	376619417292767232*(lr**10*sigma**19) + 39777841885298688*(lr**9*rBar*sigma**19) + 39635441489543168*(lr**10*sigma**20)),\
	lambda \
	sigma,lr,rBar : np.sqrt(1207959552*(lr)**12 + 60699967488*(lr**12*sigma) + 6794772480*(lr**11*rBar*sigma) + \
	1461404565504*(lr**12*sigma**2) + 312370790400*(lr**11*rBar*sigma**2) + 17543725056*(lr**10*rBar**2*sigma**2) + \
	22434319171584*(lr**12*sigma**3) + 6854217302016*(lr**11*rBar*sigma**3) + 731910242304*(lr**10*rBar**2*sigma**3) + \
	27509391360*(lr**9*rBar**3*sigma**3) + 246528852885504*(lr**12*sigma**4) + 95494039142400*(lr**11*rBar*sigma**4) + \
	14507056300032*(lr**10*rBar**2*sigma**4) + 1031361527808*(lr**9*rBar**3*sigma**4) + 29196288000*(lr**8*rBar**4*sigma**4) + \
	2063838629855232*(lr**12*sigma**5) + 947930885259264*(lr**11*rBar*sigma**5) + 181631476039680*(lr**10*rBar**2*sigma**5) + \
	18266117898240*(lr**9*rBar**3*sigma**5) + 971878957056*(lr**8*rBar**4*sigma**5) + 22111322112*(lr**7*rBar**5*sigma**5) + \
	13676550607208448*(lr**12*sigma**6) + 7131299493445632*(lr**11*rBar*sigma**6) + 1610919485374464*(lr**10*rBar**2*sigma**6) + \
	203034889617408*(lr**9*rBar**3*sigma**6) + 15172755259392*(lr**8*rBar**4*sigma**6) + 643592355840*(lr**7*rBar**5*sigma**6) + \
	12262440960*(lr**6*rBar**6*sigma**6) + 73586521460441088*(lr**12*sigma**7) + 42220589695008768*(lr**11*rBar*sigma**7) + \
	10757745504387072*(lr**10*rBar**2*sigma**7) + 1587041397964800*(lr**9*rBar**3*sigma**7) + 147438680555520*(lr**8*rBar**4*sigma**7) + \
	8703335596032*(lr**7*rBar**5*sigma**7) + 305892163584*(lr**6*rBar**6*sigma**7) + 5021761536*(lr**5*rBar**7*sigma**7) + \
	327153527916871680*(lr**12*sigma**8) + 201631668202487808*(lr**11*rBar*sigma**8) + 56121987699597312*(lr**10*rBar**2*sigma**8) + \
	9262855868129280*(lr**9*rBar**3*sigma**8) + 997954696599552*(lr**8*rBar**4*sigma**8) + 72457917585408*(lr**7*rBar**5*sigma**8) + \
	3500246237184*(lr**6*rBar**6*sigma**8) + 104444854272*(lr**5*rBar**7*sigma**8) + 1508474880*(lr**4*rBar**8*sigma**8) + \
	1216660080859545600*(lr**12*sigma**9) + 789793674108862464*(lr**11*rBar*sigma**9) + 234196439139385344*(lr**10*rBar**2*sigma**9) + \
	41831170156505088*(lr**9*rBar**3*sigma**9) + 4988178903490560*(lr**8*rBar**4*sigma**9) + 414793153723392*(lr**7*rBar**5*sigma**9) + \
	24284857089024*(lr**6*rBar**6*sigma**9) + 978462867456*(lr**5*rBar**7*sigma**9) + 25130041344*(lr**4*rBar**8*sigma**9) + \
	324403200*(lr**3*rBar**9*sigma**9) + 3817203131557085184*(lr**12*sigma**10) + 2566143979665269760*(lr**11*rBar*sigma**10) + \
	793904496838824960*(lr**10*rBar**2*sigma**10) + 149438812652356608*(lr**9*rBar**3*sigma**10) + \
	19043043169830912*(lr**8*rBar**4*sigma**10) + 1726813596727296*(lr**7*rBar**5*sigma**10) + 113741299459584*(lr**6*rBar**6*sigma**10) + \
	5434131483648*(lr**5*rBar**7*sigma**10) + 183349739520*(lr**4*rBar**8*sigma**10) + 4061380608*(lr**3*rBar**9*sigma**10) + \
	47443968*(lr**2*rBar**10*sigma**10) + 10160356382837047296*(lr**12*sigma**11) + 6966618475166348544*(lr**11*rBar*sigma**11) + \
	2207658375254981376*(lr**10*rBar**2*sigma**11) + 428072539981409280*(lr**9*rBar**3*sigma**11) + \
	56631018122237952*(lr**8*rBar**4*sigma**11) + 5390035225604352*(lr**7*rBar**5*sigma**11) + 378738397177344*(lr**6*rBar**6*sigma**11) + \
	19803941199360*(lr**5*rBar**7*sigma**11) + 764586938880*(lr**4*rBar**8*sigma**11) + 21208559616*(lr**3*rBar**9*sigma**11) + \
	397099008*(lr**2*rBar**10*sigma**11) + 4239360*(lr*rBar**11*sigma**11) + 23016650332708012032*(lr**12*sigma**12) + \
	15867635186236980096*(lr**11*rBar*sigma**12) + 5063068806599990016*(lr**10*rBar**2*sigma**12) + \
	990499741563072384*(lr**9*rBar**3*sigma**12) + 132562409432083200*(lr**8*rBar**4*sigma**12) + \
	12811908786378624*(lr**7*rBar**5*sigma**12) + 919079931495168*(lr**6*rBar**6*sigma**12) + 49464564311808*(lr**5*rBar**7*sigma**12) + \
	1991934762624*(lr**4*rBar**8*sigma**12) + 59054412672*(lr**3*rBar**9*sigma**12) + 1247477760*(lr**2*rBar**10*sigma**12) + \
	17805312*(lr*rBar**11*sigma**12) + 175104*(rBar**12*sigma**12) + 44421500673881800704*(lr**12*sigma**13) + \
	30361132234502528832*(lr**11*rBar*sigma**13) + 9592885583784016704*(lr**10*rBar**2*sigma**13) + \
	1855498401177533568*(lr**9*rBar**3*sigma**13) + 245027455987497984*(lr**8*rBar**4*sigma**13) + \
	23299154079219072*(lr**7*rBar**5*sigma**13) + 1637275602972096*(lr**6*rBar**6*sigma**13) + 85720745185920*(lr**5*rBar**7*sigma**13) + \
	3318137686656*(lr**4*rBar**8*sigma**13) + 92409845184*(lr**3*rBar**9*sigma**13) + 1739282112*(lr**2*rBar**10*sigma**13) + \
	18690048*(lr*rBar**11*sigma**13) + 72964702986571874304*(lr**12*sigma**14) + 48736127249591274720*(lr**11*rBar*sigma**14) + \
	14988357897341744736*(lr**10*rBar**2*sigma**14) + 2807148356242867584*(lr**9*rBar**3*sigma**14) + \
	356393673066321216*(lr**8*rBar**4*sigma**14) + 32252813131423104*(lr**7*rBar**5*sigma**14) + \
	2124509420233728*(lr**6*rBar**6*sigma**14) + 101745082965888*(lr**5*rBar**7*sigma**14) + 3450341934432*(lr**4*rBar**8*sigma**14) + \
	77033884608*(lr**3*rBar**9*sigma**14) + 909980352*(lr**2*rBar**10*sigma**14) + 101675901546991190016*(lr**12*sigma**15) + \
	65360388055222542240*(lr**11*rBar*sigma**15) + 19206097138011326832*(lr**10*rBar**2*sigma**15) + \
	3404065692503096352*(lr**9*rBar**3*sigma**15) + 403571424665769408*(lr**8*rBar**4*sigma**15) + \
	33448827185167824*(lr**7*rBar**5*sigma**15) + 1957847085014832*(lr**6*rBar**6*sigma**15) + 79147477891008*(lr**5*rBar**7*sigma**15) + \
	2047851745776*(lr**4*rBar**8*sigma**15) + 26764543872*(lr**3*rBar**9*sigma**15) + 119530452944550887424*(lr**12*sigma**16) + \
	72698523409477562256*(lr**11*rBar*sigma**16) + 19985741223008670432*(lr**10*rBar**2*sigma**16) + \
	3263580580353095304*(lr**9*rBar**3*sigma**16) + 348748675158456000*(lr**8*rBar**4*sigma**16) + \
	25197818304078216*(lr**7*rBar**5*sigma**16) + 1216232490390624*(lr**6*rBar**6*sigma**16) + 36438684382872*(lr**5*rBar**7*sigma**16) + \
	531627112080*(lr**4*rBar**8*sigma**16) + 117543708062776295424*(lr**12*sigma**17) + 66308798375243763552*(lr**11*rBar*sigma**17) + \
	16628299873816900968*(lr**10*rBar**2*sigma**17) + 2419202213631701280*(lr**9*rBar**3*sigma**17) + \
	222315672344607360*(lr**8*rBar**4*sigma**17) + 13033912997878488*(lr**7*rBar**5*sigma**17) + 457314396182208*(lr**6*rBar**6*sigma**17) \
	+ 7544263425624*(lr**5*rBar**7*sigma**17) + 95509561778085298176*(lr**12*sigma**18) + 48774125869771475952*(lr**11*rBar*sigma**18) + \
	10802103298131133368*(lr**10*rBar**2*sigma**18) + 1337859498415321548*(lr**9*rBar**3*sigma**18) + \
	98592879925205316*(lr**8*rBar**4*sigma**18) + 4144211573425956*(lr**7*rBar**5*sigma**18) + 78749737146000*(lr**6*rBar**6*sigma**18) + \
	63009965397120122880*(lr**12*sigma**19) + 28225833602188731624*(lr**11*rBar*sigma**19) + \
	5280423554548453824*(lr**10*rBar**2*sigma**19) + 519787357182756450*(lr**9*rBar**3*sigma**19) + \
	27180914215156587*(lr**8*rBar**4*sigma**19) + 611297467302276*(lr**7*rBar**5*sigma**19) + 32904677713426513920*(lr**12*sigma**20) + \
	12375272719815747120*(lr**11*rBar*sigma**20) + 1827323300853076086*(lr**10*rBar**2*sigma**20) + \
	126621367663012164*(lr**9*rBar**3*sigma**20) + 3510054771421311*(lr**8*rBar**4*sigma**20) + 13090355318032957440*(lr**12*sigma**21) + \
	3864496238244006048*(lr**11*rBar*sigma**21) + 399175030730717982*(lr**10*rBar**2*sigma**21) + \
	14557436723929932*(lr**9*rBar**3*sigma**21) + 3727769690891354112*(lr**12*sigma**22) + 765961668381354192*(lr**11*rBar*sigma**22) + \
	41400992051824626*(lr**10*rBar**2*sigma**22) + 676858259663290368*(lr**12*sigma**23) + 72436826809616328*(lr**11*rBar*sigma**23) + \
	58881938753323008*(lr**12*sigma**24)),\
	lambda sigma,lr,rBar : np.sqrt(4831838208*(lr)**12 + 242799869952*(lr**12*sigma) + \
	27179089920*(lr**11*rBar*sigma) + 5845618262016*(lr**12*sigma**2) + 1249483161600*(lr**11*rBar*sigma**2) + \
	70174900224*(lr**10*rBar**2*sigma**2) + 89737276686336*(lr**12*sigma**3) + 27416869208064*(lr**11*rBar*sigma**3) + \
	2927640969216*(lr**10*rBar**2*sigma**3) + 110037565440*(lr**9*rBar**3*sigma**3) + 986115411542016*(lr**12*sigma**4) + \
	381976156569600*(lr**11*rBar*sigma**4) + 58028225200128*(lr**10*rBar**2*sigma**4) + 4125446111232*(lr**9*rBar**3*sigma**4) + \
	116785152000*(lr**8*rBar**4*sigma**4) + 8255354519420928*(lr**12*sigma**5) + 3791723541037056*(lr**11*rBar*sigma**5) + \
	726525904158720*(lr**10*rBar**2*sigma**5) + 73064471592960*(lr**9*rBar**3*sigma**5) + 3887515828224*(lr**8*rBar**4*sigma**5) + \
	88445288448*(lr**7*rBar**5*sigma**5) + 54706202428833792*(lr**12*sigma**6) + 28525197973782528*(lr**11*rBar*sigma**6) + \
	6443677941497856*(lr**10*rBar**2*sigma**6) + 812139558469632*(lr**9*rBar**3*sigma**6) + 60691021037568*(lr**8*rBar**4*sigma**6) + \
	2574369423360*(lr**7*rBar**5*sigma**6) + 49049763840*(lr**6*rBar**6*sigma**6) + 294346085841764352*(lr**12*sigma**7) + \
	168882358780035072*(lr**11*rBar*sigma**7) + 43030982017548288*(lr**10*rBar**2*sigma**7) + 6348165591859200*(lr**9*rBar**3*sigma**7) + \
	589754722222080*(lr**8*rBar**4*sigma**7) + 34813342384128*(lr**7*rBar**5*sigma**7) + 1223568654336*(lr**6*rBar**6*sigma**7) + \
	20087046144*(lr**5*rBar**7*sigma**7) + 1308614111667486720*(lr**12*sigma**8) + 806526672809951232*(lr**11*rBar*sigma**8) + \
	224487950798389248*(lr**10*rBar**2*sigma**8) + 37051423472517120*(lr**9*rBar**3*sigma**8) + 3991818786398208*(lr**8*rBar**4*sigma**8) \
	+ 289831670341632*(lr**7*rBar**5*sigma**8) + 14000984948736*(lr**6*rBar**6*sigma**8) + 417779417088*(lr**5*rBar**7*sigma**8) + \
	6033899520*(lr**4*rBar**8*sigma**8) + 4866640323438182400*(lr**12*sigma**9) + 3159174696435449856*(lr**11*rBar*sigma**9) + \
	936785756557541376*(lr**10*rBar**2*sigma**9) + 167324680626020352*(lr**9*rBar**3*sigma**9) + \
	19952715613962240*(lr**8*rBar**4*sigma**9) + 1659172614893568*(lr**7*rBar**5*sigma**9) + 97139428356096*(lr**6*rBar**6*sigma**9) + \
	3913851469824*(lr**5*rBar**7*sigma**9) + 100520165376*(lr**4*rBar**8*sigma**9) + 1297612800*(lr**3*rBar**9*sigma**9) + \
	15268812526228340736*(lr**12*sigma**10) + 10264575918661079040*(lr**11*rBar*sigma**10) + \
	3175617987355299840*(lr**10*rBar**2*sigma**10) + 597755250609426432*(lr**9*rBar**3*sigma**10) + \
	76172172679323648*(lr**8*rBar**4*sigma**10) + 6907254386909184*(lr**7*rBar**5*sigma**10) + 454965197838336*(lr**6*rBar**6*sigma**10) + \
	21736525934592*(lr**5*rBar**7*sigma**10) + 733398958080*(lr**4*rBar**8*sigma**10) + 16245522432*(lr**3*rBar**9*sigma**10) + \
	189775872*(lr**2*rBar**10*sigma**10) + 40641425531348189184*(lr**12*sigma**11) + 27866473900665394176*(lr**11*rBar*sigma**11) + \
	8830633501019925504*(lr**10*rBar**2*sigma**11) + 1712290159925637120*(lr**9*rBar**3*sigma**11) + \
	226524072488951808*(lr**8*rBar**4*sigma**11) + 21560140902417408*(lr**7*rBar**5*sigma**11) + \
	1514953588709376*(lr**6*rBar**6*sigma**11) + 79215764797440*(lr**5*rBar**7*sigma**11) + 3058347755520*(lr**4*rBar**8*sigma**11) + \
	84834238464*(lr**3*rBar**9*sigma**11) + 1588396032*(lr**2*rBar**10*sigma**11) + 16957440*(lr*rBar**11*sigma**11) + \
	92066601330832048128*(lr**12*sigma**12) + 63470540744947920384*(lr**11*rBar*sigma**12) + \
	20252275226399960064*(lr**10*rBar**2*sigma**12) + 3961998966252289536*(lr**9*rBar**3*sigma**12) + \
	530249637728332800*(lr**8*rBar**4*sigma**12) + 51247635145514496*(lr**7*rBar**5*sigma**12) + \
	3676319725980672*(lr**6*rBar**6*sigma**12) + 197858257247232*(lr**5*rBar**7*sigma**12) + 7967739050496*(lr**4*rBar**8*sigma**12) + \
	236217650688*(lr**3*rBar**9*sigma**12) + 4989911040*(lr**2*rBar**10*sigma**12) + 71221248*(lr*rBar**11*sigma**12) + \
	700416*(rBar**12*sigma**12) + 177686002695527202816*(lr**12*sigma**13) + 121444528938010115328*(lr**11*rBar*sigma**13) + \
	38371542335136066816*(lr**10*rBar**2*sigma**13) + 7421993604710134272*(lr**9*rBar**3*sigma**13) + \
	980109823949991936*(lr**8*rBar**4*sigma**13) + 93196616316876288*(lr**7*rBar**5*sigma**13) + \
	6549102411888384*(lr**6*rBar**6*sigma**13) + 342882980743680*(lr**5*rBar**7*sigma**13) + 13272550746624*(lr**4*rBar**8*sigma**13) + \
	369639380736*(lr**3*rBar**9*sigma**13) + 6957128448*(lr**2*rBar**10*sigma**13) + 74760192*(lr*rBar**11*sigma**13) + \
	291858811946287497216*(lr**12*sigma**14) + 194944508998365098880*(lr**11*rBar*sigma**14) + \
	59953431589366978944*(lr**10*rBar**2*sigma**14) + 11228593424971470336*(lr**9*rBar**3*sigma**14) + \
	1425574692265284864*(lr**8*rBar**4*sigma**14) + 129011252525692416*(lr**7*rBar**5*sigma**14) + \
	8498037680934912*(lr**6*rBar**6*sigma**14) + 406980331863552*(lr**5*rBar**7*sigma**14) + 13801367737728*(lr**4*rBar**8*sigma**14) + \
	308135538432*(lr**3*rBar**9*sigma**14) + 3639921408*(lr**2*rBar**10*sigma**14) + 406703606187964760064*(lr**12*sigma**15) + \
	261441552220890168960*(lr**11*rBar*sigma**15) + 76824388552045307328*(lr**10*rBar**2*sigma**15) + \
	13616262770012385408*(lr**9*rBar**3*sigma**15) + 1614285698663077632*(lr**8*rBar**4*sigma**15) + \
	133795308740671296*(lr**7*rBar**5*sigma**15) + 7831388340059328*(lr**6*rBar**6*sigma**15) + 316589911564032*(lr**5*rBar**7*sigma**15) \
	+ 8191406983104*(lr**4*rBar**8*sigma**15) + 107058175488*(lr**3*rBar**9*sigma**15) + 478121811778203549696*(lr**12*sigma**16) + \
	290794093637910249024*(lr**11*rBar*sigma**16) + 79942964892034681728*(lr**10*rBar**2*sigma**16) + \
	13054322321412381216*(lr**9*rBar**3*sigma**16) + 1394994700633824000*(lr**8*rBar**4*sigma**16) + \
	100791273216312864*(lr**7*rBar**5*sigma**16) + 4864929961562496*(lr**6*rBar**6*sigma**16) + 145754737531488*(lr**5*rBar**7*sigma**16) \
	+ 2126508448320*(lr**4*rBar**8*sigma**16) + 470174832251105181696*(lr**12*sigma**17) + 265235193500975054208*(lr**11*rBar*sigma**17) + \
	66513199495267603872*(lr**10*rBar**2*sigma**17) + 9676808854526805120*(lr**9*rBar**3*sigma**17) + \
	889262689378429440*(lr**8*rBar**4*sigma**17) + 52135651991513952*(lr**7*rBar**5*sigma**17) + \
	1829257584728832*(lr**6*rBar**6*sigma**17) + 30177053702496*(lr**5*rBar**7*sigma**17) + 382038247112341192704*(lr**12*sigma**18) + \
	195096503479085903808*(lr**11*rBar*sigma**18) + 43208413192524533472*(lr**10*rBar**2*sigma**18) + \
	5351437993661286192*(lr**9*rBar**3*sigma**18) + 394371519700821264*(lr**8*rBar**4*sigma**18) + \
	16576846293703824*(lr**7*rBar**5*sigma**18) + 314998948584000*(lr**6*rBar**6*sigma**18) + 252039861588480491520*(lr**12*sigma**19) + \
	112903334408754926496*(lr**11*rBar*sigma**19) + 21121694218193815296*(lr**10*rBar**2*sigma**19) + \
	2079149428731025800*(lr**9*rBar**3*sigma**19) + 108723656860626348*(lr**8*rBar**4*sigma**19) + \
	2445189869209104*(lr**7*rBar**5*sigma**19) + 131618710853706055680*(lr**12*sigma**20) + 49501090879262988480*(lr**11*rBar*sigma**20) + \
	7309293203412304344*(lr**10*rBar**2*sigma**20) + 506485470652048656*(lr**9*rBar**3*sigma**20) + \
	14040219085685244*(lr**8*rBar**4*sigma**20) + 52361421272131829760*(lr**12*sigma**21) + 15457984952976024192*(lr**11*rBar*sigma**21) + \
	1596700122922871928*(lr**10*rBar**2*sigma**21) + 58229746895719728*(lr**9*rBar**3*sigma**21) + 14911078763565416448*(lr**12*sigma**22) \
	+ 3063846673525416768*(lr**11*rBar*sigma**22) + 165603968207298504*(lr**10*rBar**2*sigma**22) + 2707433038653161472*(lr**12*sigma**23) \
	+ 289747307238465312*(lr**11*rBar*sigma**23) + 235527755013292032*(lr**12*sigma**24)),\
	lambda sigma,lr,rBar : \
	np.sqrt(19327352832*(lr)**12 + 971199479808*(lr**12*sigma) + 108716359680*(lr**11*rBar*sigma) + 23382473048064*(lr**12*sigma**2) + \
	4997932646400*(lr**11*rBar*sigma**2) + 280699600896*(lr**10*rBar**2*sigma**2) + 358949106745344*(lr**12*sigma**3) + \
	109667476832256*(lr**11*rBar*sigma**3) + 11710563876864*(lr**10*rBar**2*sigma**3) + 440150261760*(lr**9*rBar**3*sigma**3) + \
	3944461646168064*(lr**12*sigma**4) + 1527904626278400*(lr**11*rBar*sigma**4) + 232112900800512*(lr**10*rBar**2*sigma**4) + \
	16501784444928*(lr**9*rBar**3*sigma**4) + 467140608000*(lr**8*rBar**4*sigma**4) + 33021418077683712*(lr**12*sigma**5) + \
	15166894164148224*(lr**11*rBar*sigma**5) + 2906103616634880*(lr**10*rBar**2*sigma**5) + 292257886371840*(lr**9*rBar**3*sigma**5) + \
	15550063312896*(lr**8*rBar**4*sigma**5) + 353781153792*(lr**7*rBar**5*sigma**5) + 218824809715335168*(lr**12*sigma**6) + \
	114100791895130112*(lr**11*rBar*sigma**6) + 25774711765991424*(lr**10*rBar**2*sigma**6) + 3248558233878528*(lr**9*rBar**3*sigma**6) + \
	242764084150272*(lr**8*rBar**4*sigma**6) + 10297477693440*(lr**7*rBar**5*sigma**6) + 196199055360*(lr**6*rBar**6*sigma**6) + \
	1177384343367057408*(lr**12*sigma**7) + 675529435120140288*(lr**11*rBar*sigma**7) + 172123928070193152*(lr**10*rBar**2*sigma**7) + \
	25392662367436800*(lr**9*rBar**3*sigma**7) + 2359018888888320*(lr**8*rBar**4*sigma**7) + 139253369536512*(lr**7*rBar**5*sigma**7) + \
	4894274617344*(lr**6*rBar**6*sigma**7) + 80348184576*(lr**5*rBar**7*sigma**7) + 5234456446669946880*(lr**12*sigma**8) + \
	3226106691239804928*(lr**11*rBar*sigma**8) + 897951803193556992*(lr**10*rBar**2*sigma**8) + \
	148205693890068480*(lr**9*rBar**3*sigma**8) + 15967275145592832*(lr**8*rBar**4*sigma**8) + 1159326681366528*(lr**7*rBar**5*sigma**8) + \
	56003939794944*(lr**6*rBar**6*sigma**8) + 1671117668352*(lr**5*rBar**7*sigma**8) + 24135598080*(lr**4*rBar**8*sigma**8) + \
	19466561293752729600*(lr**12*sigma**9) + 12636698785741799424*(lr**11*rBar*sigma**9) + 3747143026230165504*(lr**10*rBar**2*sigma**9) + \
	669298722504081408*(lr**9*rBar**3*sigma**9) + 79810862455848960*(lr**8*rBar**4*sigma**9) + 6636690459574272*(lr**7*rBar**5*sigma**9) + \
	388557713424384*(lr**6*rBar**6*sigma**9) + 15655405879296*(lr**5*rBar**7*sigma**9) + 402080661504*(lr**4*rBar**8*sigma**9) + \
	5190451200*(lr**3*rBar**9*sigma**9) + 61075250104913362944*(lr**12*sigma**10) + 41058303674644316160*(lr**11*rBar*sigma**10) + \
	12702471949421199360*(lr**10*rBar**2*sigma**10) + 2391021002437705728*(lr**9*rBar**3*sigma**10) + \
	304688690717294592*(lr**8*rBar**4*sigma**10) + 27629017547636736*(lr**7*rBar**5*sigma**10) + \
	1819860791353344*(lr**6*rBar**6*sigma**10) + 86946103738368*(lr**5*rBar**7*sigma**10) + 2933595832320*(lr**4*rBar**8*sigma**10) + \
	64982089728*(lr**3*rBar**9*sigma**10) + 759103488*(lr**2*rBar**10*sigma**10) + 162565702125392756736*(lr**12*sigma**11) + \
	111465895602661576704*(lr**11*rBar*sigma**11) + 35322534004079702016*(lr**10*rBar**2*sigma**11) + \
	6849160639702548480*(lr**9*rBar**3*sigma**11) + 906096289955807232*(lr**8*rBar**4*sigma**11) + \
	86240563609669632*(lr**7*rBar**5*sigma**11) + 6059814354837504*(lr**6*rBar**6*sigma**11) + 316863059189760*(lr**5*rBar**7*sigma**11) + \
	12233391022080*(lr**4*rBar**8*sigma**11) + 339336953856*(lr**3*rBar**9*sigma**11) + 6353584128*(lr**2*rBar**10*sigma**11) + \
	67829760*(lr*rBar**11*sigma**11) + 368266405323328192512*(lr**12*sigma**12) + 253882162979791681536*(lr**11*rBar*sigma**12) + \
	81009100905599840256*(lr**10*rBar**2*sigma**12) + 15847995865009158144*(lr**9*rBar**3*sigma**12) + \
	2120998550913331200*(lr**8*rBar**4*sigma**12) + 204990540582057984*(lr**7*rBar**5*sigma**12) + \
	14705278903922688*(lr**6*rBar**6*sigma**12) + 791433028988928*(lr**5*rBar**7*sigma**12) + 31870956201984*(lr**4*rBar**8*sigma**12) + \
	944870602752*(lr**3*rBar**9*sigma**12) + 19959644160*(lr**2*rBar**10*sigma**12) + 284884992*(lr*rBar**11*sigma**12) + \
	2801664*(rBar**12*sigma**12) + 710744010782108811264*(lr**12*sigma**13) + 485778115752040461312*(lr**11*rBar*sigma**13) + \
	153486169340544267264*(lr**10*rBar**2*sigma**13) + 29687974418840537088*(lr**9*rBar**3*sigma**13) + \
	3920439295799967744*(lr**8*rBar**4*sigma**13) + 372786465267505152*(lr**7*rBar**5*sigma**13) + \
	26196409647553536*(lr**6*rBar**6*sigma**13) + 1371531922974720*(lr**5*rBar**7*sigma**13) + 53090202986496*(lr**4*rBar**8*sigma**13) + \
	1478557522944*(lr**3*rBar**9*sigma**13) + 27828513792*(lr**2*rBar**10*sigma**13) + 299040768*(lr*rBar**11*sigma**13) + \
	1167435247785149988864*(lr**12*sigma**14) + 779778035993460395520*(lr**11*rBar*sigma**14) + \
	239813726357467915776*(lr**10*rBar**2*sigma**14) + 44914373699885881344*(lr**9*rBar**3*sigma**14) + \
	5702298769061139456*(lr**8*rBar**4*sigma**14) + 516045010102769664*(lr**7*rBar**5*sigma**14) + \
	33992150723739648*(lr**6*rBar**6*sigma**14) + 1627921327454208*(lr**5*rBar**7*sigma**14) + 55205470950912*(lr**4*rBar**8*sigma**14) + \
	1232542153728*(lr**3*rBar**9*sigma**14) + 14559685632*(lr**2*rBar**10*sigma**14) + 1626814424751859040256*(lr**12*sigma**15) + \
	1045766208883560675840*(lr**11*rBar*sigma**15) + 307297554208181229312*(lr**10*rBar**2*sigma**15) + \
	54465051080049541632*(lr**9*rBar**3*sigma**15) + 6457142794652310528*(lr**8*rBar**4*sigma**15) + \
	535181234962685184*(lr**7*rBar**5*sigma**15) + 31325553360237312*(lr**6*rBar**6*sigma**15) + \
	1266359646256128*(lr**5*rBar**7*sigma**15) + 32765627932416*(lr**4*rBar**8*sigma**15) + 428232701952*(lr**3*rBar**9*sigma**15) + \
	1912487247112814198784*(lr**12*sigma**16) + 1163176374551640996096*(lr**11*rBar*sigma**16) + \
	319771859568138726912*(lr**10*rBar**2*sigma**16) + 52217289285649524864*(lr**9*rBar**3*sigma**16) + \
	5579978802535296000*(lr**8*rBar**4*sigma**16) + 403165092865251456*(lr**7*rBar**5*sigma**16) + \
	19459719846249984*(lr**6*rBar**6*sigma**16) + 583018950125952*(lr**5*rBar**7*sigma**16) + 8506033793280*(lr**4*rBar**8*sigma**16) + \
	1880699329004420726784*(lr**12*sigma**17) + 1060940774003900216832*(lr**11*rBar*sigma**17) + \
	266052797981070415488*(lr**10*rBar**2*sigma**17) + 38707235418107220480*(lr**9*rBar**3*sigma**17) + \
	3557050757513717760*(lr**8*rBar**4*sigma**17) + 208542607966055808*(lr**7*rBar**5*sigma**17) + \
	7317030338915328*(lr**6*rBar**6*sigma**17) + 120708214809984*(lr**5*rBar**7*sigma**17) + 1528152988449364770816*(lr**12*sigma**18) + \
	780386013916343615232*(lr**11*rBar*sigma**18) + 172833652770098133888*(lr**10*rBar**2*sigma**18) + \
	21405751974645144768*(lr**9*rBar**3*sigma**18) + 1577486078803285056*(lr**8*rBar**4*sigma**18) + \
	66307385174815296*(lr**7*rBar**5*sigma**18) + 1259995794336000*(lr**6*rBar**6*sigma**18) + 1008159446353921966080*(lr**12*sigma**19) + \
	451613337635019705984*(lr**11*rBar*sigma**19) + 84486776872775261184*(lr**10*rBar**2*sigma**19) + \
	8316597714924103200*(lr**9*rBar**3*sigma**19) + 434894627442505392*(lr**8*rBar**4*sigma**19) + \
	9780759476836416*(lr**7*rBar**5*sigma**19) + 526474843414824222720*(lr**12*sigma**20) + 198004363517051953920*(lr**11*rBar*sigma**20) \
	+ 29237172813649217376*(lr**10*rBar**2*sigma**20) + 2025941882608194624*(lr**9*rBar**3*sigma**20) + \
	56160876342740976*(lr**8*rBar**4*sigma**20) + 209445685088527319040*(lr**12*sigma**21) + 61831939811904096768*(lr**11*rBar*sigma**21) \
	+ 6386800491691487712*(lr**10*rBar**2*sigma**21) + 232918987582878912*(lr**9*rBar**3*sigma**21) + \
	59644315054261665792*(lr**12*sigma**22) + 12255386694101667072*(lr**11*rBar*sigma**22) + 662415872829194016*(lr**10*rBar**2*sigma**22) \
	+ 10829732154612645888*(lr**12*sigma**23) + 1158989228953861248*(lr**11*rBar*sigma**23) + 942111020053168128*(lr**12*sigma**24)),\
	lambda sigma,lr,rBar : np.sqrt(8192*(lr)**8 + 284672*(lr**8*sigma) + 30720*(lr**7*rBar*sigma) + 4637952*(lr**8*sigma**2) + \
	924160*(lr**7*rBar*sigma**2) + 50432*(lr**6*rBar**2*sigma**2) + 47023872*(lr**8*sigma**3) + 12927232*(lr**7*rBar*sigma**3) + \
	1289472*(lr**6*rBar**2*sigma**3) + 47360*(lr**5*rBar**3*sigma**3) + 332078016*(lr**8*sigma**4) + 111423488*(lr**7*rBar*sigma**4) + \
	15153792*(lr**6*rBar**2*sigma**4) + 1003520*(lr**5*rBar**3*sigma**4) + 27840*(lr**4*rBar**4*sigma**4) + 1731959680*(lr**8*sigma**5) + \
	661027456*(lr**7*rBar*sigma**5) + 108201600*(lr**6*rBar**2*sigma**5) + 9609600*(lr**5*rBar**3*sigma**5) + \
	471040*(lr**4*rBar**4*sigma**5) + 10496*(lr**3*rBar**5*sigma**5) + 6901043952*(lr**8*sigma**6) + 2854983392*(lr**7*rBar*sigma**6) + \
	522654480*(lr**6*rBar**2*sigma**6) + 54738240*(lr**5*rBar**3*sigma**6) + 3507280*(lr**4*rBar**4*sigma**6) + \
	133536*(lr**3*rBar**5*sigma**6) + 2480*(lr**2*rBar**6*sigma**6) + 21429294960*(lr**8*sigma**7) + 9256559504*(lr**7*rBar*sigma**7) + \
	1798828560*(lr**6*rBar**2*sigma**7) + 205305840*(lr**5*rBar**3*sigma**7) + 14999440*(lr**4*rBar**4*sigma**7) + \
	713136*(lr**3*rBar**5*sigma**7) + 21200*(lr**2*rBar**6*sigma**7) + 336*(lr*rBar**7*sigma**7) + 52410449572*(lr**8*sigma**8) + \
	22885679504*(lr**7*rBar*sigma**8) + 4522564880*(lr**6*rBar**2*sigma**8) + 529620720*(lr**5*rBar**3*sigma**8) + \
	40269320*(lr**4*rBar**4*sigma**8) + 2043696*(lr**3*rBar**5*sigma**8) + 68496*(lr**2*rBar**6*sigma**8) + 1456*(lr*rBar**7*sigma**8) + \
	20*(rBar**8*sigma**8) + 101298655096*(lr**8*sigma**9) + 43356962808*(lr**7*rBar*sigma**9) + 8369142696*(lr**6*rBar**2*sigma**9) + \
	951576360*(lr**5*rBar**3*sigma**9) + 69477640*(lr**4*rBar**4*sigma**9) + 3311592*(lr**3*rBar**5*sigma**9) + \
	98904*(lr**2*rBar**6*sigma**9) + 1576*(lr*rBar**7*sigma**9) + 154219170004*(lr**8*sigma**10) + 62640269432*(lr**7*rBar*sigma**10) + \
	11314631820*(lr**6*rBar**2*sigma**10) + 1175882000*(lr**5*rBar**3*sigma**10) + 75221420*(lr**4*rBar**4*sigma**10) + \
	2875416*(lr**3*rBar**5*sigma**10) + 53796*(lr**2*rBar**6*sigma**10) + 183001304976*(lr**8*sigma**11) + \
	67946347024*(lr**7*rBar*sigma**11) + 10899751200*(lr**6*rBar**2*sigma**11) + 956436640*(lr**5*rBar**3*sigma**11) + \
	46717200*(lr**4*rBar**4*sigma**11) + 1044784*(lr**3*rBar**5*sigma**11) + 165940069296*(lr**8*sigma**12) + \
	53665001408*(lr**7*rBar*sigma**12) + 7102193952*(lr**6*rBar**2*sigma**12) + 462343360*(lr**5*rBar**3*sigma**12) + \
	12739280*(lr**4*rBar**4*sigma**12) + 111164836096*(lr**8*sigma**13) + 29177444352*(lr**7*rBar*sigma**13) + \
	2810364672*(lr**6*rBar**2*sigma**13) + 100846720*(lr**5*rBar**3*sigma**13) + 51891827392*(lr**8*sigma**14) + \
	9775262080*(lr**7*rBar*sigma**14) + 510678464*(lr**6*rBar**2*sigma**14) + 15082192128*(lr**8*sigma**15) + \
	1522549248*(lr**7*rBar*sigma**15) + 2056388864*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : np.sqrt(32768*(lr)**8 + \
	1138688*(lr**8*sigma) + 122880*(lr**7*rBar*sigma) + 18551808*(lr**8*sigma**2) + 3696640*(lr**7*rBar*sigma**2) + \
	201728*(lr**6*rBar**2*sigma**2) + 188095488*(lr**8*sigma**3) + 51708928*(lr**7*rBar*sigma**3) + 5157888*(lr**6*rBar**2*sigma**3) + \
	189440*(lr**5*rBar**3*sigma**3) + 1328312064*(lr**8*sigma**4) + 445693952*(lr**7*rBar*sigma**4) + 60615168*(lr**6*rBar**2*sigma**4) + \
	4014080*(lr**5*rBar**3*sigma**4) + 111360*(lr**4*rBar**4*sigma**4) + 6927838720*(lr**8*sigma**5) + 2644109824*(lr**7*rBar*sigma**5) + \
	432806400*(lr**6*rBar**2*sigma**5) + 38438400*(lr**5*rBar**3*sigma**5) + 1884160*(lr**4*rBar**4*sigma**5) + \
	41984*(lr**3*rBar**5*sigma**5) + 27604175808*(lr**8*sigma**6) + 11419933568*(lr**7*rBar*sigma**6) + \
	2090617920*(lr**6*rBar**2*sigma**6) + 218952960*(lr**5*rBar**3*sigma**6) + 14029120*(lr**4*rBar**4*sigma**6) + \
	534144*(lr**3*rBar**5*sigma**6) + 9920*(lr**2*rBar**6*sigma**6) + 85717179840*(lr**8*sigma**7) + 37026238016*(lr**7*rBar*sigma**7) + \
	7195314240*(lr**6*rBar**2*sigma**7) + 821223360*(lr**5*rBar**3*sigma**7) + 59997760*(lr**4*rBar**4*sigma**7) + \
	2852544*(lr**3*rBar**5*sigma**7) + 84800*(lr**2*rBar**6*sigma**7) + 1344*(lr*rBar**7*sigma**7) + 209641798288*(lr**8*sigma**8) + \
	91542718016*(lr**7*rBar*sigma**8) + 18090259520*(lr**6*rBar**2*sigma**8) + 2118482880*(lr**5*rBar**3*sigma**8) + \
	161077280*(lr**4*rBar**4*sigma**8) + 8174784*(lr**3*rBar**5*sigma**8) + 273984*(lr**2*rBar**6*sigma**8) + 5824*(lr*rBar**7*sigma**8) + \
	80*(rBar**8*sigma**8) + 405194620384*(lr**8*sigma**9) + 173427851232*(lr**7*rBar*sigma**9) + 33476570784*(lr**6*rBar**2*sigma**9) + \
	3806305440*(lr**5*rBar**3*sigma**9) + 277910560*(lr**4*rBar**4*sigma**9) + 13246368*(lr**3*rBar**5*sigma**9) + \
	395616*(lr**2*rBar**6*sigma**9) + 6304*(lr*rBar**7*sigma**9) + 616876680016*(lr**8*sigma**10) + 250561077728*(lr**7*rBar*sigma**10) + \
	45258527280*(lr**6*rBar**2*sigma**10) + 4703528000*(lr**5*rBar**3*sigma**10) + 300885680*(lr**4*rBar**4*sigma**10) + \
	11501664*(lr**3*rBar**5*sigma**10) + 215184*(lr**2*rBar**6*sigma**10) + 732005219904*(lr**8*sigma**11) + \
	271785388096*(lr**7*rBar*sigma**11) + 43599004800*(lr**6*rBar**2*sigma**11) + 3825746560*(lr**5*rBar**3*sigma**11) + \
	186868800*(lr**4*rBar**4*sigma**11) + 4179136*(lr**3*rBar**5*sigma**11) + 663760277184*(lr**8*sigma**12) + \
	214660005632*(lr**7*rBar*sigma**12) + 28408775808*(lr**6*rBar**2*sigma**12) + 1849373440*(lr**5*rBar**3*sigma**12) + \
	50957120*(lr**4*rBar**4*sigma**12) + 444659344384*(lr**8*sigma**13) + 116709777408*(lr**7*rBar*sigma**13) + \
	11241458688*(lr**6*rBar**2*sigma**13) + 403386880*(lr**5*rBar**3*sigma**13) + 207567309568*(lr**8*sigma**14) + \
	39101048320*(lr**7*rBar*sigma**14) + 2042713856*(lr**6*rBar**2*sigma**14) + 60328768512*(lr**8*sigma**15) + \
	6090196992*(lr**7*rBar*sigma**15) + 8225555456*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : np.sqrt(131072*(lr)**8 + \
	4554752*(lr**8*sigma) + 491520*(lr**7*rBar*sigma) + 74207232*(lr**8*sigma**2) + 14786560*(lr**7*rBar*sigma**2) + \
	806912*(lr**6*rBar**2*sigma**2) + 752381952*(lr**8*sigma**3) + 206835712*(lr**7*rBar*sigma**3) + 20631552*(lr**6*rBar**2*sigma**3) + \
	757760*(lr**5*rBar**3*sigma**3) + 5313248256*(lr**8*sigma**4) + 1782775808*(lr**7*rBar*sigma**4) + 242460672*(lr**6*rBar**2*sigma**4) \
	+ 16056320*(lr**5*rBar**3*sigma**4) + 445440*(lr**4*rBar**4*sigma**4) + 27711354880*(lr**8*sigma**5) + \
	10576439296*(lr**7*rBar*sigma**5) + 1731225600*(lr**6*rBar**2*sigma**5) + 153753600*(lr**5*rBar**3*sigma**5) + \
	7536640*(lr**4*rBar**4*sigma**5) + 167936*(lr**3*rBar**5*sigma**5) + 110416703232*(lr**8*sigma**6) + 45679734272*(lr**7*rBar*sigma**6) \
	+ 8362471680*(lr**6*rBar**2*sigma**6) + 875811840*(lr**5*rBar**3*sigma**6) + 56116480*(lr**4*rBar**4*sigma**6) + \
	2136576*(lr**3*rBar**5*sigma**6) + 39680*(lr**2*rBar**6*sigma**6) + 342868719360*(lr**8*sigma**7) + 148104952064*(lr**7*rBar*sigma**7) \
	+ 28781256960*(lr**6*rBar**2*sigma**7) + 3284893440*(lr**5*rBar**3*sigma**7) + 239991040*(lr**4*rBar**4*sigma**7) + \
	11410176*(lr**3*rBar**5*sigma**7) + 339200*(lr**2*rBar**6*sigma**7) + 5376*(lr*rBar**7*sigma**7) + 838567193152*(lr**8*sigma**8) + \
	366170872064*(lr**7*rBar*sigma**8) + 72361038080*(lr**6*rBar**2*sigma**8) + 8473931520*(lr**5*rBar**3*sigma**8) + \
	644309120*(lr**4*rBar**4*sigma**8) + 32699136*(lr**3*rBar**5*sigma**8) + 1095936*(lr**2*rBar**6*sigma**8) + \
	23296*(lr*rBar**7*sigma**8) + 320*(rBar**8*sigma**8) + 1620778481536*(lr**8*sigma**9) + 693711404928*(lr**7*rBar*sigma**9) + \
	133906283136*(lr**6*rBar**2*sigma**9) + 15225221760*(lr**5*rBar**3*sigma**9) + 1111642240*(lr**4*rBar**4*sigma**9) + \
	52985472*(lr**3*rBar**5*sigma**9) + 1582464*(lr**2*rBar**6*sigma**9) + 25216*(lr*rBar**7*sigma**9) + 2467506720064*(lr**8*sigma**10) + \
	1002244310912*(lr**7*rBar*sigma**10) + 181034109120*(lr**6*rBar**2*sigma**10) + 18814112000*(lr**5*rBar**3*sigma**10) + \
	1203542720*(lr**4*rBar**4*sigma**10) + 46006656*(lr**3*rBar**5*sigma**10) + 860736*(lr**2*rBar**6*sigma**10) + \
	2928020879616*(lr**8*sigma**11) + 1087141552384*(lr**7*rBar*sigma**11) + 174396019200*(lr**6*rBar**2*sigma**11) + \
	15302986240*(lr**5*rBar**3*sigma**11) + 747475200*(lr**4*rBar**4*sigma**11) + 16716544*(lr**3*rBar**5*sigma**11) + \
	2655041108736*(lr**8*sigma**12) + 858640022528*(lr**7*rBar*sigma**12) + 113635103232*(lr**6*rBar**2*sigma**12) + \
	7397493760*(lr**5*rBar**3*sigma**12) + 203828480*(lr**4*rBar**4*sigma**12) + 1778637377536*(lr**8*sigma**13) + \
	466839109632*(lr**7*rBar*sigma**13) + 44965834752*(lr**6*rBar**2*sigma**13) + 1613547520*(lr**5*rBar**3*sigma**13) + \
	830269238272*(lr**8*sigma**14) + 156404193280*(lr**7*rBar*sigma**14) + 8170855424*(lr**6*rBar**2*sigma**14) + \
	241315074048*(lr**8*sigma**15) + 24360787968*(lr**7*rBar*sigma**15) + 32902221824*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : \
	np.sqrt(524288*(lr)**8 + 18219008*(lr**8*sigma) + 1966080*(lr**7*rBar*sigma) + 296828928*(lr**8*sigma**2) + \
	59146240*(lr**7*rBar*sigma**2) + 3227648*(lr**6*rBar**2*sigma**2) + 3009527808*(lr**8*sigma**3) + 827342848*(lr**7*rBar*sigma**3) + \
	82526208*(lr**6*rBar**2*sigma**3) + 3031040*(lr**5*rBar**3*sigma**3) + 21252993024*(lr**8*sigma**4) + 7131103232*(lr**7*rBar*sigma**4) \
	+ 969842688*(lr**6*rBar**2*sigma**4) + 64225280*(lr**5*rBar**3*sigma**4) + 1781760*(lr**4*rBar**4*sigma**4) + \
	110845419520*(lr**8*sigma**5) + 42305757184*(lr**7*rBar*sigma**5) + 6924902400*(lr**6*rBar**2*sigma**5) + \
	615014400*(lr**5*rBar**3*sigma**5) + 30146560*(lr**4*rBar**4*sigma**5) + 671744*(lr**3*rBar**5*sigma**5) + \
	441666812928*(lr**8*sigma**6) + 182718937088*(lr**7*rBar*sigma**6) + 33449886720*(lr**6*rBar**2*sigma**6) + \
	3503247360*(lr**5*rBar**3*sigma**6) + 224465920*(lr**4*rBar**4*sigma**6) + 8546304*(lr**3*rBar**5*sigma**6) + \
	158720*(lr**2*rBar**6*sigma**6) + 1371474877440*(lr**8*sigma**7) + 592419808256*(lr**7*rBar*sigma**7) + \
	115125027840*(lr**6*rBar**2*sigma**7) + 13139573760*(lr**5*rBar**3*sigma**7) + 959964160*(lr**4*rBar**4*sigma**7) + \
	45640704*(lr**3*rBar**5*sigma**7) + 1356800*(lr**2*rBar**6*sigma**7) + 21504*(lr*rBar**7*sigma**7) + 3354268772608*(lr**8*sigma**8) + \
	1464683488256*(lr**7*rBar*sigma**8) + 289444152320*(lr**6*rBar**2*sigma**8) + 33895726080*(lr**5*rBar**3*sigma**8) + \
	2577236480*(lr**4*rBar**4*sigma**8) + 130796544*(lr**3*rBar**5*sigma**8) + 4383744*(lr**2*rBar**6*sigma**8) + \
	93184*(lr*rBar**7*sigma**8) + 1280*(rBar**8*sigma**8) + 6483113926144*(lr**8*sigma**9) + 2774845619712*(lr**7*rBar*sigma**9) + \
	535625132544*(lr**6*rBar**2*sigma**9) + 60900887040*(lr**5*rBar**3*sigma**9) + 4446568960*(lr**4*rBar**4*sigma**9) + \
	211941888*(lr**3*rBar**5*sigma**9) + 6329856*(lr**2*rBar**6*sigma**9) + 100864*(lr*rBar**7*sigma**9) + 9870026880256*(lr**8*sigma**10) \
	+ 4008977243648*(lr**7*rBar*sigma**10) + 724136436480*(lr**6*rBar**2*sigma**10) + 75256448000*(lr**5*rBar**3*sigma**10) + \
	4814170880*(lr**4*rBar**4*sigma**10) + 184026624*(lr**3*rBar**5*sigma**10) + 3442944*(lr**2*rBar**6*sigma**10) + \
	11712083518464*(lr**8*sigma**11) + 4348566209536*(lr**7*rBar*sigma**11) + 697584076800*(lr**6*rBar**2*sigma**11) + \
	61211944960*(lr**5*rBar**3*sigma**11) + 2989900800*(lr**4*rBar**4*sigma**11) + 66866176*(lr**3*rBar**5*sigma**11) + \
	10620164434944*(lr**8*sigma**12) + 3434560090112*(lr**7*rBar*sigma**12) + 454540412928*(lr**6*rBar**2*sigma**12) + \
	29589975040*(lr**5*rBar**3*sigma**12) + 815313920*(lr**4*rBar**4*sigma**12) + 7114549510144*(lr**8*sigma**13) + \
	1867356438528*(lr**7*rBar*sigma**13) + 179863339008*(lr**6*rBar**2*sigma**13) + 6454190080*(lr**5*rBar**3*sigma**13) + \
	3321076953088*(lr**8*sigma**14) + 625616773120*(lr**7*rBar*sigma**14) + 32683421696*(lr**6*rBar**2*sigma**14) + \
	965260296192*(lr**8*sigma**15) + 97443151872*(lr**7*rBar*sigma**15) + 131608887296*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : \
	np.sqrt(2097152*(lr)**8 + 72876032*(lr**8*sigma) + 7864320*(lr**7*rBar*sigma) + 1187315712*(lr**8*sigma**2) + \
	236584960*(lr**7*rBar*sigma**2) + 12910592*(lr**6*rBar**2*sigma**2) + 12038111232*(lr**8*sigma**3) + 3309371392*(lr**7*rBar*sigma**3) \
	+ 330104832*(lr**6*rBar**2*sigma**3) + 12124160*(lr**5*rBar**3*sigma**3) + 85011972096*(lr**8*sigma**4) + \
	28524412928*(lr**7*rBar*sigma**4) + 3879370752*(lr**6*rBar**2*sigma**4) + 256901120*(lr**5*rBar**3*sigma**4) + \
	7127040*(lr**4*rBar**4*sigma**4) + 443381678080*(lr**8*sigma**5) + 169223028736*(lr**7*rBar*sigma**5) + \
	27699609600*(lr**6*rBar**2*sigma**5) + 2460057600*(lr**5*rBar**3*sigma**5) + 120586240*(lr**4*rBar**4*sigma**5) + \
	2686976*(lr**3*rBar**5*sigma**5) + 1766667251712*(lr**8*sigma**6) + 730875748352*(lr**7*rBar*sigma**6) + \
	133799546880*(lr**6*rBar**2*sigma**6) + 14012989440*(lr**5*rBar**3*sigma**6) + 897863680*(lr**4*rBar**4*sigma**6) + \
	34185216*(lr**3*rBar**5*sigma**6) + 634880*(lr**2*rBar**6*sigma**6) + 5485899509760*(lr**8*sigma**7) + \
	2369679233024*(lr**7*rBar*sigma**7) + 460500111360*(lr**6*rBar**2*sigma**7) + 52558295040*(lr**5*rBar**3*sigma**7) + \
	3839856640*(lr**4*rBar**4*sigma**7) + 182562816*(lr**3*rBar**5*sigma**7) + 5427200*(lr**2*rBar**6*sigma**7) + \
	86016*(lr*rBar**7*sigma**7) + 13417075090432*(lr**8*sigma**8) + 5858733953024*(lr**7*rBar*sigma**8) + \
	1157776609280*(lr**6*rBar**2*sigma**8) + 135582904320*(lr**5*rBar**3*sigma**8) + 10308945920*(lr**4*rBar**4*sigma**8) + \
	523186176*(lr**3*rBar**5*sigma**8) + 17534976*(lr**2*rBar**6*sigma**8) + 372736*(lr*rBar**7*sigma**8) + 5120*(rBar**8*sigma**8) + \
	25932455704576*(lr**8*sigma**9) + 11099382478848*(lr**7*rBar*sigma**9) + 2142500530176*(lr**6*rBar**2*sigma**9) + \
	243603548160*(lr**5*rBar**3*sigma**9) + 17786275840*(lr**4*rBar**4*sigma**9) + 847767552*(lr**3*rBar**5*sigma**9) + \
	25319424*(lr**2*rBar**6*sigma**9) + 403456*(lr*rBar**7*sigma**9) + 39480107521024*(lr**8*sigma**10) + \
	16035908974592*(lr**7*rBar*sigma**10) + 2896545745920*(lr**6*rBar**2*sigma**10) + 301025792000*(lr**5*rBar**3*sigma**10) + \
	19256683520*(lr**4*rBar**4*sigma**10) + 736106496*(lr**3*rBar**5*sigma**10) + 13771776*(lr**2*rBar**6*sigma**10) + \
	46848334073856*(lr**8*sigma**11) + 17394264838144*(lr**7*rBar*sigma**11) + 2790336307200*(lr**6*rBar**2*sigma**11) + \
	244847779840*(lr**5*rBar**3*sigma**11) + 11959603200*(lr**4*rBar**4*sigma**11) + 267464704*(lr**3*rBar**5*sigma**11) + \
	42480657739776*(lr**8*sigma**12) + 13738240360448*(lr**7*rBar*sigma**12) + 1818161651712*(lr**6*rBar**2*sigma**12) + \
	118359900160*(lr**5*rBar**3*sigma**12) + 3261255680*(lr**4*rBar**4*sigma**12) + 28458198040576*(lr**8*sigma**13) + \
	7469425754112*(lr**7*rBar*sigma**13) + 719453356032*(lr**6*rBar**2*sigma**13) + 25816760320*(lr**5*rBar**3*sigma**13) + \
	13284307812352*(lr**8*sigma**14) + 2502467092480*(lr**7*rBar*sigma**14) + 130733686784*(lr**6*rBar**2*sigma**14) + \
	3861041184768*(lr**8*sigma**15) + 389772607488*(lr**7*rBar*sigma**15) + 526435549184*(lr**8*sigma**16)) \
)

deriv3DenomBoundsLambdas = [ deriv3DenomBoundsLambdas[k] for k in deriv3NonProportionalDenoms]

deriv3DenomLambdasCfuns = ( \
	"#define denom(xi,beta,sigma,lr,rBar) (pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) \
- 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", \
	"#define denom(xi,beta,sigma,lr,rBar) (2*pow(2*lr*pow(1 - sigma + \
sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), \
3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - \
4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", \
	"#\
define denom(xi,beta,sigma,lr,rBar) (4*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) \
- 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", \
	"#define denom(xi,beta,sigma,lr,rBar) (8*pow(2*lr*pow(1 - sigma + \
sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), \
3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - \
4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", \
	"#\
define denom(xi,beta,sigma,lr,rBar) (16*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) \
- 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", \
	"#define denom(xi,beta,sigma,lr,rBar) (32*pow(2*lr*pow(1 - sigma + \
sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), \
3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - \
4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", \
	"#\
define denom(xi,beta,sigma,lr,rBar) (64*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) \
- 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", \
	"#define denom(xi,beta,sigma,lr,rBar) (128*pow(2*lr*pow(1 - sigma + \
sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), \
3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - \
4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", \
	"#\
define denom(xi,beta,sigma,lr,rBar) (256*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) \
+ lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - \
2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", \
	"#define denom(xi,beta,sigma,lr,rBar) (pow(3*lr*pow(sigma, \
2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, \
2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2), 2)*pow(2*lr*pow(1 - sigma + \
sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), \
3))", \
	"#define denom(xi,beta,sigma,lr,rBar) (2*pow(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + \
sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta \
- xi/2) - 2*rBar*sigma*sin(beta + xi/2), 2)*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - \
rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))", \
	"#define \
denom(xi,beta,sigma,lr,rBar) (4*pow(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + \
sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta \
- xi/2) - 2*rBar*sigma*sin(beta + xi/2), 2)*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - \
rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))", \
	"#define \
denom(xi,beta,sigma,lr,rBar) (8*pow(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + \
sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta \
- xi/2) - 2*rBar*sigma*sin(beta + xi/2), 2)*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - \
rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))", \
	"#define \
denom(xi,beta,sigma,lr,rBar) (16*pow(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + \
sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta \
- xi/2) - 2*rBar*sigma*sin(beta + xi/2), 2)*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - \
rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))", \
	"#define \
denom(xi,beta,sigma,lr,rBar) (32*pow(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + \
sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta \
- xi/2) - 2*rBar*sigma*sin(beta + xi/2), 2)*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - \
rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))", \
	"#define \
denom(xi,beta,sigma,lr,rBar) (pow(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta \
- (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - xi/2) - \
2*rBar*sigma*sin(beta + xi/2), 3)*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))", "#define denom(xi,beta,sigma,lr,rBar) (2*pow(3*lr*pow(sigma, \
2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, \
2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2), 3)*pow(2*lr*pow(1 - sigma + \
sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), \
3))", \
	"#define denom(xi,beta,sigma,lr,rBar) (4*pow(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + \
sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta \
- xi/2) - 2*rBar*sigma*sin(beta + xi/2), 3)*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - \
rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))", \
	"#define \
denom(xi,beta,sigma,lr,rBar) (pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(-2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) + \
rBar*sigma*cos(beta)*sin(xi/2) - lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2)))", \
	"#define \
denom(xi,beta,sigma,lr,rBar) (2*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(-2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) + \
rBar*sigma*cos(beta)*sin(xi/2) - lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2)))", \
	"#define \
denom(xi,beta,sigma,lr,rBar) (4*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(-2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) + \
rBar*sigma*cos(beta)*sin(xi/2) - lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2)))", \
	"#define \
denom(xi,beta,sigma,lr,rBar) (8*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(-2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) + \
rBar*sigma*cos(beta)*sin(xi/2) - lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2)))", \
	"#define \
denom(xi,beta,sigma,lr,rBar) (16*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(-2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) + \
rBar*sigma*cos(beta)*sin(xi/2) - lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2)))")

deriv3DenomLambdasCfuns = [ deriv3DenomLambdasCfuns[k] for k in deriv3NonProportionalDenoms]

deriv2NumBoundLambda = lambda sigma,lr,rBar : 8*(lr**2*sigma) + 55*(lr**3*sigma) + 384*(lr**4*sigma) + 10*(lr*rBar*sigma) + 48*(lr**2*rBar*sigma) + \
384*(lr**3*rBar*sigma) + 43*(lr**2*sigma**2) + 524*(lr**3*sigma**2) + 5296*(lr**4*sigma**2) + 21*(lr*rBar*sigma**2) + \
347*(lr**2*rBar*sigma**2) + 4544*(lr**3*rBar*sigma**2) + (rBar**2*sigma**2) + 29*(lr*rBar**2*sigma**2) + 336*(lr**2*rBar**2*sigma**2) \
+ 387*(lr**2*sigma**3) + 3311*(lr**3*sigma**3) + 31284*(lr**4*sigma**3) + 141*(lr*rBar*sigma**3) + 1781*(lr**2*rBar*sigma**3) + \
22428*(lr**3*rBar*sigma**3) + 92*(lr*rBar**2*sigma**3) + 2564*(lr**2*rBar**2*sigma**3) + 3*(rBar**3*sigma**3) + \
84*(lr*rBar**3*sigma**3) + 261*(lr**2*sigma**4) + 9039*(lr**3*sigma**4) + 102602*(lr**4*sigma**4) + 3492*(lr**2*rBar*sigma**4) + \
59377*(lr**3*rBar*sigma**4) + 322*(lr*rBar**2*sigma**4) + 7386*(lr**2*rBar**2*sigma**4) + 320*(lr*rBar**3*sigma**4) + \
6*(rBar**4*sigma**4) + 8538*(lr**3*sigma**5) + 215272*(lr**4*sigma**5) + 2372*(lr**2*rBar*sigma**5) + 95489*(lr**3*rBar*sigma**5) + \
11357*(lr**2*rBar**2*sigma**5) + 446*(lr*rBar**3*sigma**5) + 4299*(lr**3*sigma**6) + 280552*(lr**4*sigma**6) + \
80775*(lr**3*rBar*sigma**6) + 6392*(lr**2*rBar**2*sigma**6) + 178892*(lr**4*sigma**7) + 30703*(lr**3*rBar*sigma**7) + \
56688*(lr**4*sigma**8)

deriv2NonProportionalDenoms = [0, 4, 8, 12, 18]

deriv2DenomLambdas = (\
	lambda xi,beta,sigma,lr,rBar : (2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + \
	lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : 2*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : 4*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : 32*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + \
	lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : (3*lr*sigma**2*np.sin(beta) + \
	lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) \
	+ 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : 2*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
	6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
	sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : 4*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + \
	4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta \
	+ xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - \
	xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : 8*(3*lr*sigma**2*np.sin(beta) + \
	lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) \
	+ 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : (3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
	6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
	sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))**2*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : 2*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + \
	4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta \
	+ xi/2))**2*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta \
	- xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : 4*(3*lr*sigma**2*np.sin(beta) + \
	lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) \
	+ 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))**2*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : 8*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
	6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
	sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))**2*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**2,\
	lambda xi,beta,sigma,lr,rBar : (2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + \
	lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 2*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 4*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 8*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + \
	lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 16*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 32*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : (3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + \
	4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta \
	+ xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - \
	xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 2*(3*lr*sigma**2*np.sin(beta) + \
	lr*sigma**2*np.sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) \
	+ 2*(rBar - 5*lr*(-1 + sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + \
	sigma*np.cos(xi/2))**2*np.sin(beta - xi) - rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + \
	sigma*np.cos(xi/2))*np.sin(xi/2))**3,\
	lambda xi,beta,sigma,lr,rBar : 4*(3*lr*sigma**2*np.sin(beta) + lr*sigma**2*np.sin(beta - 2*xi) - \
	6*lr*(-1 + sigma)*sigma*np.sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*sigma**2)*np.sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
	sigma))*sigma*np.sin(beta - xi/2) - 2*rBar*sigma*np.sin(beta + xi/2))*(2*lr*(1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi) - \
	rBar*sigma*np.cos(beta)*np.sin(xi/2) + lr*sigma*np.cos(beta - xi)*(1 - sigma + sigma*np.cos(xi/2))*np.sin(xi/2))**3
)

deriv2DenomLambdas = [ deriv2DenomLambdas[k] for k in deriv2NonProportionalDenoms]

deriv2DenomBoundsLambdas = (\
	lambda sigma,lr,rBar : np.sqrt(128*(lr)**4 + 2144*(lr**4*sigma) + 224*(lr**3*rBar*sigma) + 15764*(lr**4*sigma**2) + \
	2792*(lr**3*rBar*sigma**2) + 148*(lr**2*rBar**2*sigma**2) + 66468*(lr**4*sigma**3) + 14620*(lr**3*rBar*sigma**3) + \
	1244*(lr**2*rBar**2*sigma**3) + 44*(lr*rBar**3*sigma**3) + 175825*(lr**4*sigma**4) + 41152*(lr**3*rBar*sigma**4) + \
	3974*(lr**2*rBar**2*sigma**4) + 192*(lr*rBar**3*sigma**4) + 5*(rBar**4*sigma**4) + 298862*(lr**4*sigma**5) + \
	65638*(lr**3*rBar*sigma**5) + 5702*(lr**2*rBar**2*sigma**5) + 210*(lr*rBar**3*sigma**5) + 318845*(lr**4*sigma**6) + \
	56218*(lr**3*rBar*sigma**6) + 3093*(lr**2*rBar**2*sigma**6) + 195244*(lr**4*sigma**7) + 20188*(lr**3*rBar*sigma**7) + \
	52548*(lr**4*sigma**8)),\
	lambda sigma,lr,rBar : np.sqrt(512*(lr)**4 + 8576*(lr**4*sigma) + 896*(lr**3*rBar*sigma) + \
	63056*(lr**4*sigma**2) + 11168*(lr**3*rBar*sigma**2) + 592*(lr**2*rBar**2*sigma**2) + 265872*(lr**4*sigma**3) + \
	58480*(lr**3*rBar*sigma**3) + 4976*(lr**2*rBar**2*sigma**3) + 176*(lr*rBar**3*sigma**3) + 703300*(lr**4*sigma**4) + \
	164608*(lr**3*rBar*sigma**4) + 15896*(lr**2*rBar**2*sigma**4) + 768*(lr*rBar**3*sigma**4) + 20*(rBar**4*sigma**4) + \
	1195448*(lr**4*sigma**5) + 262552*(lr**3*rBar*sigma**5) + 22808*(lr**2*rBar**2*sigma**5) + 840*(lr*rBar**3*sigma**5) + \
	1275380*(lr**4*sigma**6) + 224872*(lr**3*rBar*sigma**6) + 12372*(lr**2*rBar**2*sigma**6) + 780976*(lr**4*sigma**7) + \
	80752*(lr**3*rBar*sigma**7) + 210192*(lr**4*sigma**8)),\
	lambda sigma,lr,rBar : np.sqrt(2048*(lr)**4 + 34304*(lr**4*sigma) + \
	3584*(lr**3*rBar*sigma) + 252224*(lr**4*sigma**2) + 44672*(lr**3*rBar*sigma**2) + 2368*(lr**2*rBar**2*sigma**2) + \
	1063488*(lr**4*sigma**3) + 233920*(lr**3*rBar*sigma**3) + 19904*(lr**2*rBar**2*sigma**3) + 704*(lr*rBar**3*sigma**3) + \
	2813200*(lr**4*sigma**4) + 658432*(lr**3*rBar*sigma**4) + 63584*(lr**2*rBar**2*sigma**4) + 3072*(lr*rBar**3*sigma**4) + \
	80*(rBar**4*sigma**4) + 4781792*(lr**4*sigma**5) + 1050208*(lr**3*rBar*sigma**5) + 91232*(lr**2*rBar**2*sigma**5) + \
	3360*(lr*rBar**3*sigma**5) + 5101520*(lr**4*sigma**6) + 899488*(lr**3*rBar*sigma**6) + 49488*(lr**2*rBar**2*sigma**6) + \
	3123904*(lr**4*sigma**7) + 323008*(lr**3*rBar*sigma**7) + 840768*(lr**4*sigma**8)),\
	lambda sigma,lr,rBar : np.sqrt(131072*(lr)**4 + \
	2195456*(lr**4*sigma) + 229376*(lr**3*rBar*sigma) + 16142336*(lr**4*sigma**2) + 2859008*(lr**3*rBar*sigma**2) + \
	151552*(lr**2*rBar**2*sigma**2) + 68063232*(lr**4*sigma**3) + 14970880*(lr**3*rBar*sigma**3) + 1273856*(lr**2*rBar**2*sigma**3) + \
	45056*(lr*rBar**3*sigma**3) + 180044800*(lr**4*sigma**4) + 42139648*(lr**3*rBar*sigma**4) + 4069376*(lr**2*rBar**2*sigma**4) + \
	196608*(lr*rBar**3*sigma**4) + 5120*(rBar**4*sigma**4) + 306034688*(lr**4*sigma**5) + 67213312*(lr**3*rBar*sigma**5) + \
	5838848*(lr**2*rBar**2*sigma**5) + 215040*(lr*rBar**3*sigma**5) + 326497280*(lr**4*sigma**6) + 57567232*(lr**3*rBar*sigma**6) + \
	3167232*(lr**2*rBar**2*sigma**6) + 199929856*(lr**4*sigma**7) + 20672512*(lr**3*rBar*sigma**7) + 53809152*(lr**4*sigma**8)),\
	lambda sigma,lr,rBar : np.sqrt(18432*(lr)**6 + 462336*(lr**6*sigma) + 49152*(lr**5*rBar*sigma) + 5319232*(lr**6*sigma**2) + \
	1022720*(lr**5*rBar*sigma**2) + 55040*(lr**4*rBar**2*sigma**2) + 37116672*(lr**6*sigma**3) + 9591552*(lr**5*rBar*sigma**3) + \
	914688*(lr**4*rBar**2*sigma**3) + 33280*(lr**3*rBar**3*sigma**3) + 174941664*(lr**6*sigma**4) + 53385472*(lr**5*rBar*sigma**4) + \
	6667616*(lr**4*rBar**2*sigma**4) + 415872*(lr**3*rBar**3*sigma**4) + 11520*(lr**2*rBar**4*sigma**4) + 586736544*(lr**6*sigma**5) + \
	195256000*(lr**5*rBar*sigma**5) + 27835616*(lr**4*rBar**2*sigma**5) + 2172992*(lr**3*rBar**3*sigma**5) + \
	96736*(lr**2*rBar**4*sigma**5) + 2176*(lr*rBar**5*sigma**5) + 1435810212*(lr**6*sigma**6) + 490270400*(lr**5*rBar*sigma**6) + \
	72761256*(lr**4*rBar**2*sigma**6) + 6071968*(lr**3*rBar**3*sigma**6) + 305828*(lr**2*rBar**4*sigma**6) + 9264*(lr*rBar**5*sigma**6) + \
	176*(rBar**6*sigma**6) + 2583006000*(lr**6*sigma**7) + 855736728*(lr**5*rBar*sigma**7) + 121893152*(lr**4*rBar**2*sigma**7) + \
	9560656*(lr**3*rBar**3*sigma**7) + 430600*(lr**2*rBar**4*sigma**7) + 9872*(lr*rBar**5*sigma**7) + 3390328944*(lr**6*sigma**8) + \
	1025068608*(lr**5*rBar*sigma**8) + 127747032*(lr**4*rBar**2*sigma**8) + 8035408*(lr**3*rBar**3*sigma**8) + \
	227928*(lr**2*rBar**4*sigma**8) + 3166305408*(lr**6*sigma**9) + 806362888*(lr**5*rBar*sigma**9) + 76545028*(lr**4*rBar**2*sigma**9) + \
	2818032*(lr**3*rBar**3*sigma**9) + 1997208000*(lr**6*sigma**10) + 376091624*(lr**5*rBar*sigma**10) + \
	20078736*(lr**4*rBar**2*sigma**10) + 763953920*(lr**6*sigma**11) + 78973056*(lr**5*rBar*sigma**11) + 134015232*(lr**6*sigma**12)), \
	lambda sigma,lr,rBar : np.sqrt(73728*(lr)**6 + 1849344*(lr**6*sigma) + 196608*(lr**5*rBar*sigma) + 21276928*(lr**6*sigma**2) + \
	4090880*(lr**5*rBar*sigma**2) + 220160*(lr**4*rBar**2*sigma**2) + 148466688*(lr**6*sigma**3) + 38366208*(lr**5*rBar*sigma**3) + \
	3658752*(lr**4*rBar**2*sigma**3) + 133120*(lr**3*rBar**3*sigma**3) + 699766656*(lr**6*sigma**4) + 213541888*(lr**5*rBar*sigma**4) + \
	26670464*(lr**4*rBar**2*sigma**4) + 1663488*(lr**3*rBar**3*sigma**4) + 46080*(lr**2*rBar**4*sigma**4) + 2346946176*(lr**6*sigma**5) + \
	781024000*(lr**5*rBar*sigma**5) + 111342464*(lr**4*rBar**2*sigma**5) + 8691968*(lr**3*rBar**3*sigma**5) + \
	386944*(lr**2*rBar**4*sigma**5) + 8704*(lr*rBar**5*sigma**5) + 5743240848*(lr**6*sigma**6) + 1961081600*(lr**5*rBar*sigma**6) + \
	291045024*(lr**4*rBar**2*sigma**6) + 24287872*(lr**3*rBar**3*sigma**6) + 1223312*(lr**2*rBar**4*sigma**6) + \
	37056*(lr*rBar**5*sigma**6) + 704*(rBar**6*sigma**6) + 10332024000*(lr**6*sigma**7) + 3422946912*(lr**5*rBar*sigma**7) + \
	487572608*(lr**4*rBar**2*sigma**7) + 38242624*(lr**3*rBar**3*sigma**7) + 1722400*(lr**2*rBar**4*sigma**7) + \
	39488*(lr*rBar**5*sigma**7) + 13561315776*(lr**6*sigma**8) + 4100274432*(lr**5*rBar*sigma**8) + 510988128*(lr**4*rBar**2*sigma**8) + \
	32141632*(lr**3*rBar**3*sigma**8) + 911712*(lr**2*rBar**4*sigma**8) + 12665221632*(lr**6*sigma**9) + 3225451552*(lr**5*rBar*sigma**9) \
	+ 306180112*(lr**4*rBar**2*sigma**9) + 11272128*(lr**3*rBar**3*sigma**9) + 7988832000*(lr**6*sigma**10) + \
	1504366496*(lr**5*rBar*sigma**10) + 80314944*(lr**4*rBar**2*sigma**10) + 3055815680*(lr**6*sigma**11) + \
	315892224*(lr**5*rBar*sigma**11) + 536060928*(lr**6*sigma**12)),\
	lambda sigma,lr,rBar : np.sqrt(294912*(lr)**6 + \
	7397376*(lr**6*sigma) + 786432*(lr**5*rBar*sigma) + 85107712*(lr**6*sigma**2) + 16363520*(lr**5*rBar*sigma**2) + \
	880640*(lr**4*rBar**2*sigma**2) + 593866752*(lr**6*sigma**3) + 153464832*(lr**5*rBar*sigma**3) + 14635008*(lr**4*rBar**2*sigma**3) + \
	532480*(lr**3*rBar**3*sigma**3) + 2799066624*(lr**6*sigma**4) + 854167552*(lr**5*rBar*sigma**4) + 106681856*(lr**4*rBar**2*sigma**4) + \
	6653952*(lr**3*rBar**3*sigma**4) + 184320*(lr**2*rBar**4*sigma**4) + 9387784704*(lr**6*sigma**5) + 3124096000*(lr**5*rBar*sigma**5) + \
	445369856*(lr**4*rBar**2*sigma**5) + 34767872*(lr**3*rBar**3*sigma**5) + 1547776*(lr**2*rBar**4*sigma**5) + \
	34816*(lr*rBar**5*sigma**5) + 22972963392*(lr**6*sigma**6) + 7844326400*(lr**5*rBar*sigma**6) + 1164180096*(lr**4*rBar**2*sigma**6) + \
	97151488*(lr**3*rBar**3*sigma**6) + 4893248*(lr**2*rBar**4*sigma**6) + 148224*(lr*rBar**5*sigma**6) + 2816*(rBar**6*sigma**6) + \
	41328096000*(lr**6*sigma**7) + 13691787648*(lr**5*rBar*sigma**7) + 1950290432*(lr**4*rBar**2*sigma**7) + \
	152970496*(lr**3*rBar**3*sigma**7) + 6889600*(lr**2*rBar**4*sigma**7) + 157952*(lr*rBar**5*sigma**7) + 54245263104*(lr**6*sigma**8) + \
	16401097728*(lr**5*rBar*sigma**8) + 2043952512*(lr**4*rBar**2*sigma**8) + 128566528*(lr**3*rBar**3*sigma**8) + \
	3646848*(lr**2*rBar**4*sigma**8) + 50660886528*(lr**6*sigma**9) + 12901806208*(lr**5*rBar*sigma**9) + \
	1224720448*(lr**4*rBar**2*sigma**9) + 45088512*(lr**3*rBar**3*sigma**9) + 31955328000*(lr**6*sigma**10) + \
	6017465984*(lr**5*rBar*sigma**10) + 321259776*(lr**4*rBar**2*sigma**10) + 12223262720*(lr**6*sigma**11) + \
	1263568896*(lr**5*rBar*sigma**11) + 2144243712*(lr**6*sigma**12)),\
	lambda sigma,lr,rBar : np.sqrt(1179648*(lr)**6 + \
	29589504*(lr**6*sigma) + 3145728*(lr**5*rBar*sigma) + 340430848*(lr**6*sigma**2) + 65454080*(lr**5*rBar*sigma**2) + \
	3522560*(lr**4*rBar**2*sigma**2) + 2375467008*(lr**6*sigma**3) + 613859328*(lr**5*rBar*sigma**3) + 58540032*(lr**4*rBar**2*sigma**3) + \
	2129920*(lr**3*rBar**3*sigma**3) + 11196266496*(lr**6*sigma**4) + 3416670208*(lr**5*rBar*sigma**4) + \
	426727424*(lr**4*rBar**2*sigma**4) + 26615808*(lr**3*rBar**3*sigma**4) + 737280*(lr**2*rBar**4*sigma**4) + \
	37551138816*(lr**6*sigma**5) + 12496384000*(lr**5*rBar*sigma**5) + 1781479424*(lr**4*rBar**2*sigma**5) + \
	139071488*(lr**3*rBar**3*sigma**5) + 6191104*(lr**2*rBar**4*sigma**5) + 139264*(lr*rBar**5*sigma**5) + 91891853568*(lr**6*sigma**6) + \
	31377305600*(lr**5*rBar*sigma**6) + 4656720384*(lr**4*rBar**2*sigma**6) + 388605952*(lr**3*rBar**3*sigma**6) + \
	19572992*(lr**2*rBar**4*sigma**6) + 592896*(lr*rBar**5*sigma**6) + 11264*(rBar**6*sigma**6) + 165312384000*(lr**6*sigma**7) + \
	54767150592*(lr**5*rBar*sigma**7) + 7801161728*(lr**4*rBar**2*sigma**7) + 611881984*(lr**3*rBar**3*sigma**7) + \
	27558400*(lr**2*rBar**4*sigma**7) + 631808*(lr*rBar**5*sigma**7) + 216981052416*(lr**6*sigma**8) + 65604390912*(lr**5*rBar*sigma**8) + \
	8175810048*(lr**4*rBar**2*sigma**8) + 514266112*(lr**3*rBar**3*sigma**8) + 14587392*(lr**2*rBar**4*sigma**8) + \
	202643546112*(lr**6*sigma**9) + 51607224832*(lr**5*rBar*sigma**9) + 4898881792*(lr**4*rBar**2*sigma**9) + \
	180354048*(lr**3*rBar**3*sigma**9) + 127821312000*(lr**6*sigma**10) + 24069863936*(lr**5*rBar*sigma**10) + \
	1285039104*(lr**4*rBar**2*sigma**10) + 48893050880*(lr**6*sigma**11) + 5054275584*(lr**5*rBar*sigma**11) + \
	8576974848*(lr**6*sigma**12)),\
	lambda sigma,lr,rBar : np.sqrt(2097152*(lr)**8 + 69730304*(lr**8*sigma) + 7602176*(lr**7*rBar*sigma) \
	+ 1086980096*(lr**8*sigma**2) + 220659712*(lr**7*rBar*sigma**2) + 12107776*(lr**6*rBar**2*sigma**2) + 10544513024*(lr**8*sigma**3) + \
	2974990336*(lr**7*rBar*sigma**3) + 300761088*(lr**6*rBar**2*sigma**3) + 11091968*(lr**5*rBar**3*sigma**3) + \
	71245877248*(lr**8*sigma**4) + 24691728384*(lr**7*rBar*sigma**4) + 3426775040*(lr**6*rBar**2*sigma**4) + \
	229498880*(lr**5*rBar**3*sigma**4) + 6410240*(lr**4*rBar**4*sigma**4) + 355520868352*(lr**8*sigma**5) + \
	140931244032*(lr**7*rBar*sigma**5) + 23675760640*(lr**6*rBar**2*sigma**5) + 2139086848*(lr**5*rBar**3*sigma**5) + \
	106209280*(lr**4*rBar**4*sigma**5) + 2400256*(lr**3*rBar**5*sigma**5) + 1355307844608*(lr**8*sigma**6) + \
	585107764224*(lr**7*rBar*sigma**6) + 110451812864*(lr**6*rBar**2*sigma**6) + 11822731776*(lr**5*rBar**3*sigma**6) + \
	770947072*(lr**4*rBar**4*sigma**6) + 29908992*(lr**3*rBar**5*sigma**6) + 570368*(lr**2*rBar**6*sigma**6) + \
	4026264326144*(lr**8*sigma**7) + 1822045001472*(lr**7*rBar*sigma**7) + 366463796992*(lr**6*rBar**2*sigma**7) + \
	42893090560*(lr**5*rBar**3*sigma**7) + 3199679232*(lr**4*rBar**4*sigma**7) + 155536384*(lr**3*rBar**5*sigma**7) + \
	4760576*(lr**2*rBar**6*sigma**7) + 78848*(lr*rBar**7*sigma**7) + 9419818192896*(lr**8*sigma**8) + 4322898363264*(lr**7*rBar*sigma**8) \
	+ 886486335616*(lr**6*rBar**2*sigma**8) + 106692795392*(lr**5*rBar**3*sigma**8) + 8299088896*(lr**4*rBar**4*sigma**8) + \
	431441792*(lr**3*rBar**5*sigma**8) + 14924544*(lr**2*rBar**6*sigma**8) + 331264*(lr*rBar**7*sigma**8) + 4864*(rBar**8*sigma**8) + \
	17413989793792*(lr**8*sigma**9) + 7851614763264*(lr**7*rBar*sigma**9) + 1575000540416*(lr**6*rBar**2*sigma**9) + \
	184198062976*(lr**5*rBar**3*sigma**9) + 13766455680*(lr**4*rBar**4*sigma**9) + 672676160*(lr**3*rBar**5*sigma**9) + \
	20766016*(lr**2*rBar**6*sigma**9) + 348160*(lr*rBar**7*sigma**9) + 25352530591744*(lr**8*sigma**10) + \
	10863185561344*(lr**7*rBar*sigma**10) + 2039326719904*(lr**6*rBar**2*sigma**10) + 217867277696*(lr**5*rBar**3*sigma**10) + \
	14255895008*(lr**4*rBar**4*sigma**10) + 558722208*(lr**3*rBar**5*sigma**10) + 10849088*(lr**2*rBar**6*sigma**10) + \
	28761971359744*(lr**8*sigma**11) + 11269226364608*(lr**7*rBar*sigma**11) + 1876387699392*(lr**6*rBar**2*sigma**11) + \
	168920005440*(lr**5*rBar**3*sigma**11) + 8426364560*(lr**4*rBar**4*sigma**11) + 193456224*(lr**3*rBar**5*sigma**11) + \
	24926310498304*(lr**8*sigma**12) + 8499102719576*(lr**7*rBar*sigma**12) + 1164361758560*(lr**6*rBar**2*sigma**12) + \
	77520814816*(lr**5*rBar**3*sigma**12) + 2178404016*(lr**4*rBar**4*sigma**12) + 15952768925696*(lr**8*sigma**13) + \
	4404981158592*(lr**7*rBar*sigma**13) + 437513156340*(lr**6*rBar**2*sigma**13) + 16000243936*(lr**5*rBar**3*sigma**13) + \
	7110523092992*(lr**8*sigma**14) + 1404339724304*(lr**7*rBar*sigma**14) + 75299261304*(lr**6*rBar**2*sigma**14) + \
	1972078903296*(lr**8*sigma**15) + 207789364848*(lr**7*rBar*sigma**15) + 256390463488*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : \
	np.sqrt(8388608*(lr)**8 + 278921216*(lr**8*sigma) + 30408704*(lr**7*rBar*sigma) + 4347920384*(lr**8*sigma**2) + \
	882638848*(lr**7*rBar*sigma**2) + 48431104*(lr**6*rBar**2*sigma**2) + 42178052096*(lr**8*sigma**3) + 11899961344*(lr**7*rBar*sigma**3) \
	+ 1203044352*(lr**6*rBar**2*sigma**3) + 44367872*(lr**5*rBar**3*sigma**3) + 284983508992*(lr**8*sigma**4) + \
	98766913536*(lr**7*rBar*sigma**4) + 13707100160*(lr**6*rBar**2*sigma**4) + 917995520*(lr**5*rBar**3*sigma**4) + \
	25640960*(lr**4*rBar**4*sigma**4) + 1422083473408*(lr**8*sigma**5) + 563724976128*(lr**7*rBar*sigma**5) + \
	94703042560*(lr**6*rBar**2*sigma**5) + 8556347392*(lr**5*rBar**3*sigma**5) + 424837120*(lr**4*rBar**4*sigma**5) + \
	9601024*(lr**3*rBar**5*sigma**5) + 5421231378432*(lr**8*sigma**6) + 2340431056896*(lr**7*rBar*sigma**6) + \
	441807251456*(lr**6*rBar**2*sigma**6) + 47290927104*(lr**5*rBar**3*sigma**6) + 3083788288*(lr**4*rBar**4*sigma**6) + \
	119635968*(lr**3*rBar**5*sigma**6) + 2281472*(lr**2*rBar**6*sigma**6) + 16105057304576*(lr**8*sigma**7) + \
	7288180005888*(lr**7*rBar*sigma**7) + 1465855187968*(lr**6*rBar**2*sigma**7) + 171572362240*(lr**5*rBar**3*sigma**7) + \
	12798716928*(lr**4*rBar**4*sigma**7) + 622145536*(lr**3*rBar**5*sigma**7) + 19042304*(lr**2*rBar**6*sigma**7) + \
	315392*(lr*rBar**7*sigma**7) + 37679272771584*(lr**8*sigma**8) + 17291593453056*(lr**7*rBar*sigma**8) + \
	3545945342464*(lr**6*rBar**2*sigma**8) + 426771181568*(lr**5*rBar**3*sigma**8) + 33196355584*(lr**4*rBar**4*sigma**8) + \
	1725767168*(lr**3*rBar**5*sigma**8) + 59698176*(lr**2*rBar**6*sigma**8) + 1325056*(lr*rBar**7*sigma**8) + 19456*(rBar**8*sigma**8) + \
	69655959175168*(lr**8*sigma**9) + 31406459053056*(lr**7*rBar*sigma**9) + 6300002161664*(lr**6*rBar**2*sigma**9) + \
	736792251904*(lr**5*rBar**3*sigma**9) + 55065822720*(lr**4*rBar**4*sigma**9) + 2690704640*(lr**3*rBar**5*sigma**9) + \
	83064064*(lr**2*rBar**6*sigma**9) + 1392640*(lr*rBar**7*sigma**9) + 101410122366976*(lr**8*sigma**10) + \
	43452742245376*(lr**7*rBar*sigma**10) + 8157306879616*(lr**6*rBar**2*sigma**10) + 871469110784*(lr**5*rBar**3*sigma**10) + \
	57023580032*(lr**4*rBar**4*sigma**10) + 2234888832*(lr**3*rBar**5*sigma**10) + 43396352*(lr**2*rBar**6*sigma**10) + \
	115047885438976*(lr**8*sigma**11) + 45076905458432*(lr**7*rBar*sigma**11) + 7505550797568*(lr**6*rBar**2*sigma**11) + \
	675680021760*(lr**5*rBar**3*sigma**11) + 33705458240*(lr**4*rBar**4*sigma**11) + 773824896*(lr**3*rBar**5*sigma**11) + \
	99705241993216*(lr**8*sigma**12) + 33996410878304*(lr**7*rBar*sigma**12) + 4657447034240*(lr**6*rBar**2*sigma**12) + \
	310083259264*(lr**5*rBar**3*sigma**12) + 8713616064*(lr**4*rBar**4*sigma**12) + 63811075702784*(lr**8*sigma**13) + \
	17619924634368*(lr**7*rBar*sigma**13) + 1750052625360*(lr**6*rBar**2*sigma**13) + 64000975744*(lr**5*rBar**3*sigma**13) + \
	28442092371968*(lr**8*sigma**14) + 5617358897216*(lr**7*rBar*sigma**14) + 301197045216*(lr**6*rBar**2*sigma**14) + \
	7888315613184*(lr**8*sigma**15) + 831157459392*(lr**7*rBar*sigma**15) + 1025561853952*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : \
	np.sqrt(33554432*(lr)**8 + 1115684864*(lr**8*sigma) + 121634816*(lr**7*rBar*sigma) + 17391681536*(lr**8*sigma**2) + \
	3530555392*(lr**7*rBar*sigma**2) + 193724416*(lr**6*rBar**2*sigma**2) + 168712208384*(lr**8*sigma**3) + \
	47599845376*(lr**7*rBar*sigma**3) + 4812177408*(lr**6*rBar**2*sigma**3) + 177471488*(lr**5*rBar**3*sigma**3) + \
	1139934035968*(lr**8*sigma**4) + 395067654144*(lr**7*rBar*sigma**4) + 54828400640*(lr**6*rBar**2*sigma**4) + \
	3671982080*(lr**5*rBar**3*sigma**4) + 102563840*(lr**4*rBar**4*sigma**4) + 5688333893632*(lr**8*sigma**5) + \
	2254899904512*(lr**7*rBar*sigma**5) + 378812170240*(lr**6*rBar**2*sigma**5) + 34225389568*(lr**5*rBar**3*sigma**5) + \
	1699348480*(lr**4*rBar**4*sigma**5) + 38404096*(lr**3*rBar**5*sigma**5) + 21684925513728*(lr**8*sigma**6) + \
	9361724227584*(lr**7*rBar*sigma**6) + 1767229005824*(lr**6*rBar**2*sigma**6) + 189163708416*(lr**5*rBar**3*sigma**6) + \
	12335153152*(lr**4*rBar**4*sigma**6) + 478543872*(lr**3*rBar**5*sigma**6) + 9125888*(lr**2*rBar**6*sigma**6) + \
	64420229218304*(lr**8*sigma**7) + 29152720023552*(lr**7*rBar*sigma**7) + 5863420751872*(lr**6*rBar**2*sigma**7) + \
	686289448960*(lr**5*rBar**3*sigma**7) + 51194867712*(lr**4*rBar**4*sigma**7) + 2488582144*(lr**3*rBar**5*sigma**7) + \
	76169216*(lr**2*rBar**6*sigma**7) + 1261568*(lr*rBar**7*sigma**7) + 150717091086336*(lr**8*sigma**8) + \
	69166373812224*(lr**7*rBar*sigma**8) + 14183781369856*(lr**6*rBar**2*sigma**8) + 1707084726272*(lr**5*rBar**3*sigma**8) + \
	132785422336*(lr**4*rBar**4*sigma**8) + 6903068672*(lr**3*rBar**5*sigma**8) + 238792704*(lr**2*rBar**6*sigma**8) + \
	5300224*(lr*rBar**7*sigma**8) + 77824*(rBar**8*sigma**8) + 278623836700672*(lr**8*sigma**9) + 125625836212224*(lr**7*rBar*sigma**9) + \
	25200008646656*(lr**6*rBar**2*sigma**9) + 2947169007616*(lr**5*rBar**3*sigma**9) + 220263290880*(lr**4*rBar**4*sigma**9) + \
	10762818560*(lr**3*rBar**5*sigma**9) + 332256256*(lr**2*rBar**6*sigma**9) + 5570560*(lr*rBar**7*sigma**9) + \
	405640489467904*(lr**8*sigma**10) + 173810968981504*(lr**7*rBar*sigma**10) + 32629227518464*(lr**6*rBar**2*sigma**10) + \
	3485876443136*(lr**5*rBar**3*sigma**10) + 228094320128*(lr**4*rBar**4*sigma**10) + 8939555328*(lr**3*rBar**5*sigma**10) + \
	173585408*(lr**2*rBar**6*sigma**10) + 460191541755904*(lr**8*sigma**11) + 180307621833728*(lr**7*rBar*sigma**11) + \
	30022203190272*(lr**6*rBar**2*sigma**11) + 2702720087040*(lr**5*rBar**3*sigma**11) + 134821832960*(lr**4*rBar**4*sigma**11) + \
	3095299584*(lr**3*rBar**5*sigma**11) + 398820967972864*(lr**8*sigma**12) + 135985643513216*(lr**7*rBar*sigma**12) + \
	18629788136960*(lr**6*rBar**2*sigma**12) + 1240333037056*(lr**5*rBar**3*sigma**12) + 34854464256*(lr**4*rBar**4*sigma**12) + \
	255244302811136*(lr**8*sigma**13) + 70479698537472*(lr**7*rBar*sigma**13) + 7000210501440*(lr**6*rBar**2*sigma**13) + \
	256003902976*(lr**5*rBar**3*sigma**13) + 113768369487872*(lr**8*sigma**14) + 22469435588864*(lr**7*rBar*sigma**14) + \
	1204788180864*(lr**6*rBar**2*sigma**14) + 31553262452736*(lr**8*sigma**15) + 3324629837568*(lr**7*rBar*sigma**15) + \
	4102247415808*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : np.sqrt(134217728*(lr)**8 + 4462739456*(lr**8*sigma) + \
	486539264*(lr**7*rBar*sigma) + 69566726144*(lr**8*sigma**2) + 14122221568*(lr**7*rBar*sigma**2) + 774897664*(lr**6*rBar**2*sigma**2) + \
	674848833536*(lr**8*sigma**3) + 190399381504*(lr**7*rBar*sigma**3) + 19248709632*(lr**6*rBar**2*sigma**3) + \
	709885952*(lr**5*rBar**3*sigma**3) + 4559736143872*(lr**8*sigma**4) + 1580270616576*(lr**7*rBar*sigma**4) + \
	219313602560*(lr**6*rBar**2*sigma**4) + 14687928320*(lr**5*rBar**3*sigma**4) + 410255360*(lr**4*rBar**4*sigma**4) + \
	22753335574528*(lr**8*sigma**5) + 9019599618048*(lr**7*rBar*sigma**5) + 1515248680960*(lr**6*rBar**2*sigma**5) + \
	136901558272*(lr**5*rBar**3*sigma**5) + 6797393920*(lr**4*rBar**4*sigma**5) + 153616384*(lr**3*rBar**5*sigma**5) + \
	86739702054912*(lr**8*sigma**6) + 37446896910336*(lr**7*rBar*sigma**6) + 7068916023296*(lr**6*rBar**2*sigma**6) + \
	756654833664*(lr**5*rBar**3*sigma**6) + 49340612608*(lr**4*rBar**4*sigma**6) + 1914175488*(lr**3*rBar**5*sigma**6) + \
	36503552*(lr**2*rBar**6*sigma**6) + 257680916873216*(lr**8*sigma**7) + 116610880094208*(lr**7*rBar*sigma**7) + \
	23453683007488*(lr**6*rBar**2*sigma**7) + 2745157795840*(lr**5*rBar**3*sigma**7) + 204779470848*(lr**4*rBar**4*sigma**7) + \
	9954328576*(lr**3*rBar**5*sigma**7) + 304676864*(lr**2*rBar**6*sigma**7) + 5046272*(lr*rBar**7*sigma**7) + \
	602868364345344*(lr**8*sigma**8) + 276665495248896*(lr**7*rBar*sigma**8) + 56735125479424*(lr**6*rBar**2*sigma**8) + \
	6828338905088*(lr**5*rBar**3*sigma**8) + 531141689344*(lr**4*rBar**4*sigma**8) + 27612274688*(lr**3*rBar**5*sigma**8) + \
	955170816*(lr**2*rBar**6*sigma**8) + 21200896*(lr*rBar**7*sigma**8) + 311296*(rBar**8*sigma**8) + 1114495346802688*(lr**8*sigma**9) + \
	502503344848896*(lr**7*rBar*sigma**9) + 100800034586624*(lr**6*rBar**2*sigma**9) + 11788676030464*(lr**5*rBar**3*sigma**9) + \
	881053163520*(lr**4*rBar**4*sigma**9) + 43051274240*(lr**3*rBar**5*sigma**9) + 1329025024*(lr**2*rBar**6*sigma**9) + \
	22282240*(lr*rBar**7*sigma**9) + 1622561957871616*(lr**8*sigma**10) + 695243875926016*(lr**7*rBar*sigma**10) + \
	130516910073856*(lr**6*rBar**2*sigma**10) + 13943505772544*(lr**5*rBar**3*sigma**10) + 912377280512*(lr**4*rBar**4*sigma**10) + \
	35758221312*(lr**3*rBar**5*sigma**10) + 694341632*(lr**2*rBar**6*sigma**10) + 1840766167023616*(lr**8*sigma**11) + \
	721230487334912*(lr**7*rBar*sigma**11) + 120088812761088*(lr**6*rBar**2*sigma**11) + 10810880348160*(lr**5*rBar**3*sigma**11) + \
	539287331840*(lr**4*rBar**4*sigma**11) + 12381198336*(lr**3*rBar**5*sigma**11) + 1595283871891456*(lr**8*sigma**12) + \
	543942574052864*(lr**7*rBar*sigma**12) + 74519152547840*(lr**6*rBar**2*sigma**12) + 4961332148224*(lr**5*rBar**3*sigma**12) + \
	139417857024*(lr**4*rBar**4*sigma**12) + 1020977211244544*(lr**8*sigma**13) + 281918794149888*(lr**7*rBar*sigma**13) + \
	28000842005760*(lr**6*rBar**2*sigma**13) + 1024015611904*(lr**5*rBar**3*sigma**13) + 455073477951488*(lr**8*sigma**14) + \
	89877742355456*(lr**7*rBar*sigma**14) + 4819152723456*(lr**6*rBar**2*sigma**14) + 126213049810944*(lr**8*sigma**15) + \
	13298519350272*(lr**7*rBar*sigma**15) + 16408989663232*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : np.sqrt(1152*(lr)**6 + \
	29664*(lr**6*sigma) + 3168*(lr**5*rBar*sigma) + 350388*(lr**6*sigma**2) + 67176*(lr**5*rBar*sigma**2) + 3636*(lr**4*rBar**2*sigma**2) \
	+ 2510352*(lr**6*sigma**3) + 642816*(lr**5*rBar*sigma**3) + 61272*(lr**4*rBar**2*sigma**3) + 2232*(lr**3*rBar**3*sigma**3) + \
	12149946*(lr**6*sigma**4) + 3654720*(lr**5*rBar*sigma**4) + 454068*(lr**4*rBar**2*sigma**4) + 28224*(lr**3*rBar**3*sigma**4) + \
	774*(lr**2*rBar**4*sigma**4) + 41851692*(lr**6*sigma**5) + 13669416*(lr**5*rBar*sigma**5) + 1931868*(lr**4*rBar**2*sigma**5) + \
	149868*(lr**3*rBar**3*sigma**5) + 6588*(lr**2*rBar**4*sigma**5) + 144*(lr*rBar**5*sigma**5) + (420842493*(lr**6*sigma**6))/4 + \
	(70276653*(lr**5*rBar*sigma**6))/2 + (20634939*(lr**4*rBar**2*sigma**6))/4 + 427275*(lr**3*rBar**3*sigma**6) + \
	(84915*(lr**2*rBar**4*sigma**6))/4 + (1251*(lr*rBar**5*sigma**6))/2 + (45*(rBar**6*sigma**6))/4 + (389000763*(lr**6*sigma**7))/2 + \
	(125723529*(lr**5*rBar*sigma**7))/2 + 8849493*(lr**4*rBar**2*sigma**7) + 689067*(lr**3*rBar**3*sigma**7) + \
	(61227*(lr**2*rBar**4*sigma**7))/2 + (1359*(lr*rBar**5*sigma**7))/2 + (1049804505*(lr**6*sigma**8))/4 + 77274009*(lr**5*rBar*sigma**8) \
	+ (19043379*(lr**4*rBar**2*sigma**8))/2 + 595629*(lr**3*rBar**3*sigma**8) + (66573*(lr**2*rBar**4*sigma**8))/4 + \
	252110430*(lr**6*sigma**9) + 62469234*(lr**5*rBar*sigma**9) + 5875434*(lr**4*rBar**2*sigma**9) + 215550*(lr**3*rBar**3*sigma**9) + \
	163664010*(lr**6*sigma**10) + 29995524*(lr**5*rBar*sigma**10) + 1592154*(lr**4*rBar**2*sigma**10) + 64475280*(lr**6*sigma**11) + \
	6498000*(lr**5*rBar*sigma**11) + 11658276*(lr**6*sigma**12)),\
	lambda sigma,lr,rBar : np.sqrt(4608*(lr)**6 + 118656*(lr**6*sigma) + \
	12672*(lr**5*rBar*sigma) + 1401552*(lr**6*sigma**2) + 268704*(lr**5*rBar*sigma**2) + 14544*(lr**4*rBar**2*sigma**2) + \
	10041408*(lr**6*sigma**3) + 2571264*(lr**5*rBar*sigma**3) + 245088*(lr**4*rBar**2*sigma**3) + 8928*(lr**3*rBar**3*sigma**3) + \
	48599784*(lr**6*sigma**4) + 14618880*(lr**5*rBar*sigma**4) + 1816272*(lr**4*rBar**2*sigma**4) + 112896*(lr**3*rBar**3*sigma**4) + \
	3096*(lr**2*rBar**4*sigma**4) + 167406768*(lr**6*sigma**5) + 54677664*(lr**5*rBar*sigma**5) + 7727472*(lr**4*rBar**2*sigma**5) + \
	599472*(lr**3*rBar**3*sigma**5) + 26352*(lr**2*rBar**4*sigma**5) + 576*(lr*rBar**5*sigma**5) + 420842493*(lr**6*sigma**6) + \
	140553306*(lr**5*rBar*sigma**6) + 20634939*(lr**4*rBar**2*sigma**6) + 1709100*(lr**3*rBar**3*sigma**6) + \
	84915*(lr**2*rBar**4*sigma**6) + 2502*(lr*rBar**5*sigma**6) + 45*(rBar**6*sigma**6) + 778001526*(lr**6*sigma**7) + \
	251447058*(lr**5*rBar*sigma**7) + 35397972*(lr**4*rBar**2*sigma**7) + 2756268*(lr**3*rBar**3*sigma**7) + \
	122454*(lr**2*rBar**4*sigma**7) + 2718*(lr*rBar**5*sigma**7) + 1049804505*(lr**6*sigma**8) + 309096036*(lr**5*rBar*sigma**8) + \
	38086758*(lr**4*rBar**2*sigma**8) + 2382516*(lr**3*rBar**3*sigma**8) + 66573*(lr**2*rBar**4*sigma**8) + 1008441720*(lr**6*sigma**9) + \
	249876936*(lr**5*rBar*sigma**9) + 23501736*(lr**4*rBar**2*sigma**9) + 862200*(lr**3*rBar**3*sigma**9) + 654656040*(lr**6*sigma**10) + \
	119982096*(lr**5*rBar*sigma**10) + 6368616*(lr**4*rBar**2*sigma**10) + 257901120*(lr**6*sigma**11) + 25992000*(lr**5*rBar*sigma**11) + \
	46633104*(lr**6*sigma**12)),\
	lambda sigma,lr,rBar : np.sqrt(18432*(lr)**6 + 474624*(lr**6*sigma) + 50688*(lr**5*rBar*sigma) + \
	5606208*(lr**6*sigma**2) + 1074816*(lr**5*rBar*sigma**2) + 58176*(lr**4*rBar**2*sigma**2) + 40165632*(lr**6*sigma**3) + \
	10285056*(lr**5*rBar*sigma**3) + 980352*(lr**4*rBar**2*sigma**3) + 35712*(lr**3*rBar**3*sigma**3) + 194399136*(lr**6*sigma**4) + \
	58475520*(lr**5*rBar*sigma**4) + 7265088*(lr**4*rBar**2*sigma**4) + 451584*(lr**3*rBar**3*sigma**4) + 12384*(lr**2*rBar**4*sigma**4) + \
	669627072*(lr**6*sigma**5) + 218710656*(lr**5*rBar*sigma**5) + 30909888*(lr**4*rBar**2*sigma**5) + 2397888*(lr**3*rBar**3*sigma**5) + \
	105408*(lr**2*rBar**4*sigma**5) + 2304*(lr*rBar**5*sigma**5) + 1683369972*(lr**6*sigma**6) + 562213224*(lr**5*rBar*sigma**6) + \
	82539756*(lr**4*rBar**2*sigma**6) + 6836400*(lr**3*rBar**3*sigma**6) + 339660*(lr**2*rBar**4*sigma**6) + 10008*(lr*rBar**5*sigma**6) + \
	180*(rBar**6*sigma**6) + 3112006104*(lr**6*sigma**7) + 1005788232*(lr**5*rBar*sigma**7) + 141591888*(lr**4*rBar**2*sigma**7) + \
	11025072*(lr**3*rBar**3*sigma**7) + 489816*(lr**2*rBar**4*sigma**7) + 10872*(lr*rBar**5*sigma**7) + 4199218020*(lr**6*sigma**8) + \
	1236384144*(lr**5*rBar*sigma**8) + 152347032*(lr**4*rBar**2*sigma**8) + 9530064*(lr**3*rBar**3*sigma**8) + \
	266292*(lr**2*rBar**4*sigma**8) + 4033766880*(lr**6*sigma**9) + 999507744*(lr**5*rBar*sigma**9) + 94006944*(lr**4*rBar**2*sigma**9) + \
	3448800*(lr**3*rBar**3*sigma**9) + 2618624160*(lr**6*sigma**10) + 479928384*(lr**5*rBar*sigma**10) + \
	25474464*(lr**4*rBar**2*sigma**10) + 1031604480*(lr**6*sigma**11) + 103968000*(lr**5*rBar*sigma**11) + 186532416*(lr**6*sigma**12)), 
	 lambda sigma,lr,rBar : np.sqrt(73728*(lr)**6 + 1898496*(lr**6*sigma) + 202752*(lr**5*rBar*sigma) + 22424832*(lr**6*sigma**2) + \
	4299264*(lr**5*rBar*sigma**2) + 232704*(lr**4*rBar**2*sigma**2) + 160662528*(lr**6*sigma**3) + 41140224*(lr**5*rBar*sigma**3) + \
	3921408*(lr**4*rBar**2*sigma**3) + 142848*(lr**3*rBar**3*sigma**3) + 777596544*(lr**6*sigma**4) + 233902080*(lr**5*rBar*sigma**4) + \
	29060352*(lr**4*rBar**2*sigma**4) + 1806336*(lr**3*rBar**3*sigma**4) + 49536*(lr**2*rBar**4*sigma**4) + 2678508288*(lr**6*sigma**5) + \
	874842624*(lr**5*rBar*sigma**5) + 123639552*(lr**4*rBar**2*sigma**5) + 9591552*(lr**3*rBar**3*sigma**5) + \
	421632*(lr**2*rBar**4*sigma**5) + 9216*(lr*rBar**5*sigma**5) + 6733479888*(lr**6*sigma**6) + 2248852896*(lr**5*rBar*sigma**6) + \
	330159024*(lr**4*rBar**2*sigma**6) + 27345600*(lr**3*rBar**3*sigma**6) + 1358640*(lr**2*rBar**4*sigma**6) + \
	40032*(lr*rBar**5*sigma**6) + 720*(rBar**6*sigma**6) + 12448024416*(lr**6*sigma**7) + 4023152928*(lr**5*rBar*sigma**7) + \
	566367552*(lr**4*rBar**2*sigma**7) + 44100288*(lr**3*rBar**3*sigma**7) + 1959264*(lr**2*rBar**4*sigma**7) + \
	43488*(lr*rBar**5*sigma**7) + 16796872080*(lr**6*sigma**8) + 4945536576*(lr**5*rBar*sigma**8) + 609388128*(lr**4*rBar**2*sigma**8) + \
	38120256*(lr**3*rBar**3*sigma**8) + 1065168*(lr**2*rBar**4*sigma**8) + 16135067520*(lr**6*sigma**9) + 3998030976*(lr**5*rBar*sigma**9) \
	+ 376027776*(lr**4*rBar**2*sigma**9) + 13795200*(lr**3*rBar**3*sigma**9) + 10474496640*(lr**6*sigma**10) + \
	1919713536*(lr**5*rBar*sigma**10) + 101897856*(lr**4*rBar**2*sigma**10) + 4126417920*(lr**6*sigma**11) + \
	415872000*(lr**5*rBar*sigma**11) + 746129664*(lr**6*sigma**12)),\
	lambda sigma,lr,rBar : np.sqrt(294912*(lr)**6 + \
	7593984*(lr**6*sigma) + 811008*(lr**5*rBar*sigma) + 89699328*(lr**6*sigma**2) + 17197056*(lr**5*rBar*sigma**2) + \
	930816*(lr**4*rBar**2*sigma**2) + 642650112*(lr**6*sigma**3) + 164560896*(lr**5*rBar*sigma**3) + 15685632*(lr**4*rBar**2*sigma**3) + \
	571392*(lr**3*rBar**3*sigma**3) + 3110386176*(lr**6*sigma**4) + 935608320*(lr**5*rBar*sigma**4) + 116241408*(lr**4*rBar**2*sigma**4) + \
	7225344*(lr**3*rBar**3*sigma**4) + 198144*(lr**2*rBar**4*sigma**4) + 10714033152*(lr**6*sigma**5) + 3499370496*(lr**5*rBar*sigma**5) + \
	494558208*(lr**4*rBar**2*sigma**5) + 38366208*(lr**3*rBar**3*sigma**5) + 1686528*(lr**2*rBar**4*sigma**5) + \
	36864*(lr*rBar**5*sigma**5) + 26933919552*(lr**6*sigma**6) + 8995411584*(lr**5*rBar*sigma**6) + 1320636096*(lr**4*rBar**2*sigma**6) + \
	109382400*(lr**3*rBar**3*sigma**6) + 5434560*(lr**2*rBar**4*sigma**6) + 160128*(lr*rBar**5*sigma**6) + 2880*(rBar**6*sigma**6) + \
	49792097664*(lr**6*sigma**7) + 16092611712*(lr**5*rBar*sigma**7) + 2265470208*(lr**4*rBar**2*sigma**7) + \
	176401152*(lr**3*rBar**3*sigma**7) + 7837056*(lr**2*rBar**4*sigma**7) + 173952*(lr*rBar**5*sigma**7) + 67187488320*(lr**6*sigma**8) + \
	19782146304*(lr**5*rBar*sigma**8) + 2437552512*(lr**4*rBar**2*sigma**8) + 152481024*(lr**3*rBar**3*sigma**8) + \
	4260672*(lr**2*rBar**4*sigma**8) + 64540270080*(lr**6*sigma**9) + 15992123904*(lr**5*rBar*sigma**9) + \
	1504111104*(lr**4*rBar**2*sigma**9) + 55180800*(lr**3*rBar**3*sigma**9) + 41897986560*(lr**6*sigma**10) + \
	7678854144*(lr**5*rBar*sigma**10) + 407591424*(lr**4*rBar**2*sigma**10) + 16505671680*(lr**6*sigma**11) + \
	1663488000*(lr**5*rBar*sigma**11) + 2984518656*(lr**6*sigma**12)),\
	lambda sigma,lr,rBar : np.sqrt(1179648*(lr)**6 + \
	30375936*(lr**6*sigma) + 3244032*(lr**5*rBar*sigma) + 358797312*(lr**6*sigma**2) + 68788224*(lr**5*rBar*sigma**2) + \
	3723264*(lr**4*rBar**2*sigma**2) + 2570600448*(lr**6*sigma**3) + 658243584*(lr**5*rBar*sigma**3) + 62742528*(lr**4*rBar**2*sigma**3) + \
	2285568*(lr**3*rBar**3*sigma**3) + 12441544704*(lr**6*sigma**4) + 3742433280*(lr**5*rBar*sigma**4) + \
	464965632*(lr**4*rBar**2*sigma**4) + 28901376*(lr**3*rBar**3*sigma**4) + 792576*(lr**2*rBar**4*sigma**4) + \
	42856132608*(lr**6*sigma**5) + 13997481984*(lr**5*rBar*sigma**5) + 1978232832*(lr**4*rBar**2*sigma**5) + \
	153464832*(lr**3*rBar**3*sigma**5) + 6746112*(lr**2*rBar**4*sigma**5) + 147456*(lr*rBar**5*sigma**5) + 107735678208*(lr**6*sigma**6) + \
	35981646336*(lr**5*rBar*sigma**6) + 5282544384*(lr**4*rBar**2*sigma**6) + 437529600*(lr**3*rBar**3*sigma**6) + \
	21738240*(lr**2*rBar**4*sigma**6) + 640512*(lr*rBar**5*sigma**6) + 11520*(rBar**6*sigma**6) + 199168390656*(lr**6*sigma**7) + \
	64370446848*(lr**5*rBar*sigma**7) + 9061880832*(lr**4*rBar**2*sigma**7) + 705604608*(lr**3*rBar**3*sigma**7) + \
	31348224*(lr**2*rBar**4*sigma**7) + 695808*(lr*rBar**5*sigma**7) + 268749953280*(lr**6*sigma**8) + 79128585216*(lr**5*rBar*sigma**8) + \
	9750210048*(lr**4*rBar**2*sigma**8) + 609924096*(lr**3*rBar**3*sigma**8) + 17042688*(lr**2*rBar**4*sigma**8) + \
	258161080320*(lr**6*sigma**9) + 63968495616*(lr**5*rBar*sigma**9) + 6016444416*(lr**4*rBar**2*sigma**9) + \
	220723200*(lr**3*rBar**3*sigma**9) + 167591946240*(lr**6*sigma**10) + 30715416576*(lr**5*rBar*sigma**10) + \
	1630365696*(lr**4*rBar**2*sigma**10) + 66022686720*(lr**6*sigma**11) + 6653952000*(lr**5*rBar*sigma**11) + \
	11938074624*(lr**6*sigma**12)),\
	lambda sigma,lr,rBar : np.sqrt(131072*(lr)**8 + 4456448*(lr**8*sigma) + 483328*(lr**7*rBar*sigma) + \
	71037952*(lr**8*sigma**2) + 14295040*(lr**7*rBar*sigma**2) + 782080*(lr**6*rBar**2*sigma**2) + 704709632*(lr**8*sigma**3) + \
	196466944*(lr**7*rBar*sigma**3) + 19737600*(lr**6*rBar**2*sigma**3) + 726272*(lr**5*rBar**3*sigma**3) + 4869364544*(lr**8*sigma**4) + \
	1662951552*(lr**7*rBar*sigma**4) + 228665088*(lr**6*rBar**2*sigma**4) + 15234560*(lr**5*rBar**3*sigma**4) + \
	424000*(lr**4*rBar**4*sigma**4) + 24849784704*(lr**8*sigma**5) + 9683804928*(lr**7*rBar*sigma**5) + \
	1607803200*(lr**6*rBar**2*sigma**5) + 144137728*(lr**5*rBar**3*sigma**5) + 7114240*(lr**4*rBar**4*sigma**5) + \
	159616*(lr**3*rBar**5*sigma**5) + 96885695072*(lr**8*sigma**6) + 41037340128*(lr**7*rBar*sigma**6) + \
	7640245968*(lr**6*rBar**2*sigma**6) + 809777024*(lr**5*rBar**3*sigma**6) + 52384576*(lr**4*rBar**4*sigma**6) + \
	2014464*(lr**3*rBar**5*sigma**6) + 37904*(lr**2*rBar**6*sigma**6) + 294381962528*(lr**8*sigma**7) + 130500123280*(lr**7*rBar*sigma**7) \
	+ 25845371808*(lr**6*rBar**2*sigma**7) + 2990706704*(lr**5*rBar**3*sigma**7) + 220985856*(lr**4*rBar**4*sigma**7) + \
	10632528*(lr**3*rBar**5*sigma**7) + 320960*(lr**2*rBar**6*sigma**7) + 5200*(lr*rBar**7*sigma**7) + 704475796036*(lr**8*sigma**8) + \
	316337408848*(lr**7*rBar*sigma**8) + 63808283968*(lr**6*rBar**2*sigma**8) + 7584777168*(lr**5*rBar**3*sigma**8) + \
	583848552*(lr**4*rBar**4*sigma**8) + 30015384*(lr**3*rBar**5*sigma**8) + 1022944*(lr**2*rBar**6*sigma**8) + \
	22240*(lr*rBar**7*sigma**8) + 316*(rBar**8*sigma**8) + 1332197931584*(lr**8*sigma**9) + 587334747088*(lr**7*rBar*sigma**9) + \
	115825266624*(lr**6*rBar**2*sigma**9) + 13373306008*(lr**5*rBar**3*sigma**9) + 988741776*(lr**4*rBar**4*sigma**9) + \
	47754264*(lr**3*rBar**5*sigma**9) + 1451996*(lr**2*rBar**6*sigma**9) + 23776*(lr*rBar**7*sigma**9) + 1984157440448*(lr**8*sigma**10) + \
	831144788448*(lr**7*rBar*sigma**10) + 153399703812*(lr**6*rBar**2*sigma**10) + 16183024348*(lr**5*rBar**3*sigma**10) + \
	1047678620*(lr**4*rBar**4*sigma**10) + 40569804*(lr**3*rBar**5*sigma**10) + 774584*(lr**2*rBar**6*sigma**10) + \
	2303035589376*(lr**8*sigma**11) + 882400410816*(lr**7*rBar*sigma**11) + 144546255456*(lr**6*rBar**2*sigma**11) + \
	12860211530*(lr**5*rBar**3*sigma**11) + 634851899*(lr**4*rBar**4*sigma**11) + 14385956*(lr**3*rBar**5*sigma**11) + \
	2042281601408*(lr**8*sigma**12) + 681517613136*(lr**7*rBar*sigma**12) + 91977843504*(lr**6*rBar**2*sigma**12) + \
	6059286268*(lr**5*rBar**3*sigma**12) + 168462959*(lr**4*rBar**4*sigma**12) + 1337591708672*(lr**8*sigma**13) + \
	361976983840*(lr**7*rBar*sigma**13) + 35484422136*(lr**6*rBar**2*sigma**13) + 1285399660*(lr**5*rBar**3*sigma**13) + \
	610210004992*(lr**8*sigma**14) + 118344729280*(lr**7*rBar*sigma**14) + 6276608548*(lr**6*rBar**2*sigma**14) + \
	173244531968*(lr**8*sigma**15) + 17968872520*(lr**7*rBar*sigma**15) + 23060538368*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : \
	np.sqrt(524288*(lr)**8 + 17825792*(lr**8*sigma) + 1933312*(lr**7*rBar*sigma) + 284151808*(lr**8*sigma**2) + \
	57180160*(lr**7*rBar*sigma**2) + 3128320*(lr**6*rBar**2*sigma**2) + 2818838528*(lr**8*sigma**3) + 785867776*(lr**7*rBar*sigma**3) + \
	78950400*(lr**6*rBar**2*sigma**3) + 2905088*(lr**5*rBar**3*sigma**3) + 19477458176*(lr**8*sigma**4) + 6651806208*(lr**7*rBar*sigma**4) \
	+ 914660352*(lr**6*rBar**2*sigma**4) + 60938240*(lr**5*rBar**3*sigma**4) + 1696000*(lr**4*rBar**4*sigma**4) + \
	99399138816*(lr**8*sigma**5) + 38735219712*(lr**7*rBar*sigma**5) + 6431212800*(lr**6*rBar**2*sigma**5) + \
	576550912*(lr**5*rBar**3*sigma**5) + 28456960*(lr**4*rBar**4*sigma**5) + 638464*(lr**3*rBar**5*sigma**5) + \
	387542780288*(lr**8*sigma**6) + 164149360512*(lr**7*rBar*sigma**6) + 30560983872*(lr**6*rBar**2*sigma**6) + \
	3239108096*(lr**5*rBar**3*sigma**6) + 209538304*(lr**4*rBar**4*sigma**6) + 8057856*(lr**3*rBar**5*sigma**6) + \
	151616*(lr**2*rBar**6*sigma**6) + 1177527850112*(lr**8*sigma**7) + 522000493120*(lr**7*rBar*sigma**7) + \
	103381487232*(lr**6*rBar**2*sigma**7) + 11962826816*(lr**5*rBar**3*sigma**7) + 883943424*(lr**4*rBar**4*sigma**7) + \
	42530112*(lr**3*rBar**5*sigma**7) + 1283840*(lr**2*rBar**6*sigma**7) + 20800*(lr*rBar**7*sigma**7) + 2817903184144*(lr**8*sigma**8) + \
	1265349635392*(lr**7*rBar*sigma**8) + 255233135872*(lr**6*rBar**2*sigma**8) + 30339108672*(lr**5*rBar**3*sigma**8) + \
	2335394208*(lr**4*rBar**4*sigma**8) + 120061536*(lr**3*rBar**5*sigma**8) + 4091776*(lr**2*rBar**6*sigma**8) + \
	88960*(lr*rBar**7*sigma**8) + 1264*(rBar**8*sigma**8) + 5328791726336*(lr**8*sigma**9) + 2349338988352*(lr**7*rBar*sigma**9) + \
	463301066496*(lr**6*rBar**2*sigma**9) + 53493224032*(lr**5*rBar**3*sigma**9) + 3954967104*(lr**4*rBar**4*sigma**9) + \
	191017056*(lr**3*rBar**5*sigma**9) + 5807984*(lr**2*rBar**6*sigma**9) + 95104*(lr*rBar**7*sigma**9) + 7936629761792*(lr**8*sigma**10) \
	+ 3324579153792*(lr**7*rBar*sigma**10) + 613598815248*(lr**6*rBar**2*sigma**10) + 64732097392*(lr**5*rBar**3*sigma**10) + \
	4190714480*(lr**4*rBar**4*sigma**10) + 162279216*(lr**3*rBar**5*sigma**10) + 3098336*(lr**2*rBar**6*sigma**10) + \
	9212142357504*(lr**8*sigma**11) + 3529601643264*(lr**7*rBar*sigma**11) + 578185021824*(lr**6*rBar**2*sigma**11) + \
	51440846120*(lr**5*rBar**3*sigma**11) + 2539407596*(lr**4*rBar**4*sigma**11) + 57543824*(lr**3*rBar**5*sigma**11) + \
	8169126405632*(lr**8*sigma**12) + 2726070452544*(lr**7*rBar*sigma**12) + 367911374016*(lr**6*rBar**2*sigma**12) + \
	24237145072*(lr**5*rBar**3*sigma**12) + 673851836*(lr**4*rBar**4*sigma**12) + 5350366834688*(lr**8*sigma**13) + \
	1447907935360*(lr**7*rBar*sigma**13) + 141937688544*(lr**6*rBar**2*sigma**13) + 5141598640*(lr**5*rBar**3*sigma**13) + \
	2440840019968*(lr**8*sigma**14) + 473378917120*(lr**7*rBar*sigma**14) + 25106434192*(lr**6*rBar**2*sigma**14) + \
	692978127872*(lr**8*sigma**15) + 71875490080*(lr**7*rBar*sigma**15) + 92242153472*(lr**8*sigma**16)),\
	lambda sigma,lr,rBar : \
	np.sqrt(2097152*(lr)**8 + 71303168*(lr**8*sigma) + 7733248*(lr**7*rBar*sigma) + 1136607232*(lr**8*sigma**2) + \
	228720640*(lr**7*rBar*sigma**2) + 12513280*(lr**6*rBar**2*sigma**2) + 11275354112*(lr**8*sigma**3) + 3143471104*(lr**7*rBar*sigma**3) \
	+ 315801600*(lr**6*rBar**2*sigma**3) + 11620352*(lr**5*rBar**3*sigma**3) + 77909832704*(lr**8*sigma**4) + \
	26607224832*(lr**7*rBar*sigma**4) + 3658641408*(lr**6*rBar**2*sigma**4) + 243752960*(lr**5*rBar**3*sigma**4) + \
	6784000*(lr**4*rBar**4*sigma**4) + 397596555264*(lr**8*sigma**5) + 154940878848*(lr**7*rBar*sigma**5) + \
	25724851200*(lr**6*rBar**2*sigma**5) + 2306203648*(lr**5*rBar**3*sigma**5) + 113827840*(lr**4*rBar**4*sigma**5) + \
	2553856*(lr**3*rBar**5*sigma**5) + 1550171121152*(lr**8*sigma**6) + 656597442048*(lr**7*rBar*sigma**6) + \
	122243935488*(lr**6*rBar**2*sigma**6) + 12956432384*(lr**5*rBar**3*sigma**6) + 838153216*(lr**4*rBar**4*sigma**6) + \
	32231424*(lr**3*rBar**5*sigma**6) + 606464*(lr**2*rBar**6*sigma**6) + 4710111400448*(lr**8*sigma**7) + \
	2088001972480*(lr**7*rBar*sigma**7) + 413525948928*(lr**6*rBar**2*sigma**7) + 47851307264*(lr**5*rBar**3*sigma**7) + \
	3535773696*(lr**4*rBar**4*sigma**7) + 170120448*(lr**3*rBar**5*sigma**7) + 5135360*(lr**2*rBar**6*sigma**7) + \
	83200*(lr*rBar**7*sigma**7) + 11271612736576*(lr**8*sigma**8) + 5061398541568*(lr**7*rBar*sigma**8) + \
	1020932543488*(lr**6*rBar**2*sigma**8) + 121356434688*(lr**5*rBar**3*sigma**8) + 9341576832*(lr**4*rBar**4*sigma**8) + \
	480246144*(lr**3*rBar**5*sigma**8) + 16367104*(lr**2*rBar**6*sigma**8) + 355840*(lr*rBar**7*sigma**8) + 5056*(rBar**8*sigma**8) + \
	21315166905344*(lr**8*sigma**9) + 9397355953408*(lr**7*rBar*sigma**9) + 1853204265984*(lr**6*rBar**2*sigma**9) + \
	213972896128*(lr**5*rBar**3*sigma**9) + 15819868416*(lr**4*rBar**4*sigma**9) + 764068224*(lr**3*rBar**5*sigma**9) + \
	23231936*(lr**2*rBar**6*sigma**9) + 380416*(lr*rBar**7*sigma**9) + 31746519047168*(lr**8*sigma**10) + \
	13298316615168*(lr**7*rBar*sigma**10) + 2454395260992*(lr**6*rBar**2*sigma**10) + 258928389568*(lr**5*rBar**3*sigma**10) + \
	16762857920*(lr**4*rBar**4*sigma**10) + 649116864*(lr**3*rBar**5*sigma**10) + 12393344*(lr**2*rBar**6*sigma**10) + \
	36848569430016*(lr**8*sigma**11) + 14118406573056*(lr**7*rBar*sigma**11) + 2312740087296*(lr**6*rBar**2*sigma**11) + \
	205763384480*(lr**5*rBar**3*sigma**11) + 10157630384*(lr**4*rBar**4*sigma**11) + 230175296*(lr**3*rBar**5*sigma**11) + \
	32676505622528*(lr**8*sigma**12) + 10904281810176*(lr**7*rBar*sigma**12) + 1471645496064*(lr**6*rBar**2*sigma**12) + \
	96948580288*(lr**5*rBar**3*sigma**12) + 2695407344*(lr**4*rBar**4*sigma**12) + 21401467338752*(lr**8*sigma**13) + \
	5791631741440*(lr**7*rBar*sigma**13) + 567750754176*(lr**6*rBar**2*sigma**13) + 20566394560*(lr**5*rBar**3*sigma**13) + \
	9763360079872*(lr**8*sigma**14) + 1893515668480*(lr**7*rBar*sigma**14) + 100425736768*(lr**6*rBar**2*sigma**14) + \
	2771912511488*(lr**8*sigma**15) + 287501960320*(lr**7*rBar*sigma**15) + 368968613888*(lr**8*sigma**16))
)

deriv2DenomBoundsLambdas = [ deriv2DenomBoundsLambdas[k] for k in deriv2NonProportionalDenoms]

deriv2DenomLambdasCfuns = ("#define denom(xi,beta,sigma,lr,rBar) (pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 2))", "#define denom(xi,beta,sigma,lr,rBar) (2*pow(2*lr*pow(1 - sigma \
+ sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + \
sigma*cos(xi/2))*sin(xi/2), 2))", "#define denom(xi,beta,sigma,lr,rBar) (4*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) \
- rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 2))", "#define \
denom(xi,beta,sigma,lr,rBar) (32*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 2))", "#define denom(xi,beta,sigma,lr,rBar) (pow(2*lr*pow(1 - sigma + \
sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), \
2)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - \
4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", "#\
define denom(xi,beta,sigma,lr,rBar) (2*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 2)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) \
- 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", "#define denom(xi,beta,sigma,lr,rBar) (4*pow(2*lr*pow(1 - sigma + \
sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), \
2)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - \
4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", "#\
define denom(xi,beta,sigma,lr,rBar) (8*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 2)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) \
- 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", "#define denom(xi,beta,sigma,lr,rBar) (pow(3*lr*pow(sigma, \
2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, \
2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2), 2)*pow(2*lr*pow(1 - sigma + \
sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), \
2))", "#define denom(xi,beta,sigma,lr,rBar) (2*pow(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + \
sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta \
- xi/2) - 2*rBar*sigma*sin(beta + xi/2), 2)*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - \
rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 2))", "#define \
denom(xi,beta,sigma,lr,rBar) (4*pow(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + \
sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta \
- xi/2) - 2*rBar*sigma*sin(beta + xi/2), 2)*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - \
rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 2))", "#define \
denom(xi,beta,sigma,lr,rBar) (8*pow(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + \
sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta \
- xi/2) - 2*rBar*sigma*sin(beta + xi/2), 2)*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - \
rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 2))", "#define \
denom(xi,beta,sigma,lr,rBar) (pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))", "#define denom(xi,beta,sigma,lr,rBar) (2*pow(2*lr*pow(1 - sigma \
+ sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + \
sigma*cos(xi/2))*sin(xi/2), 3))", "#define denom(xi,beta,sigma,lr,rBar) (4*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) \
- rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))", "#define \
denom(xi,beta,sigma,lr,rBar) (8*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))", "#define denom(xi,beta,sigma,lr,rBar) (16*pow(2*lr*pow(1 - \
sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + \
sigma*cos(xi/2))*sin(xi/2), 3))", "#define denom(xi,beta,sigma,lr,rBar) (32*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - \
xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))", "#define \
denom(xi,beta,sigma,lr,rBar) (pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) \
- 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", "#define denom(xi,beta,sigma,lr,rBar) (2*pow(2*lr*pow(1 - sigma + \
sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), \
3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - \
4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))", "#\
define denom(xi,beta,sigma,lr,rBar) (4*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) + \
lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3)*(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) \
- 6*lr*(-1 + sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + \
sigma))*sigma*sin(beta - xi/2) - 2*rBar*sigma*sin(beta + xi/2)))")

deriv2DenomLambdasCfuns = [ deriv2DenomLambdasCfuns[k] for k in deriv2NonProportionalDenoms]









deriv1NumBoundLambda = lambda sigma,lr,rBar : 1


deriv1DenomLambdas = [ \
		lambda xi,beta,sigma,lr,rBar : -(((1 - sigma + sigma*np.cos(xi/2))**2*np.sin(beta - xi))/rBar**2) + \
			(sigma*np.cos(beta)*np.sin(xi/2))/(2*lr*rBar) - (sigma*np.cos(beta - xi)*(1 - sigma + \
			sigma*np.cos(xi/2))*np.sin(xi/2))/(2*rBar**2)
	]

deriv1DenomBoundsLambdas = [ \
	lambda sigma,lr,rBar : np.sqrt(2/(rBar)**4 + (31*(sigma/rBar**4))/2 + (3*(sigma/(lr*rBar**3)))/2 + (741*(sigma**2/rBar**4))/16 \
		+ (53*(sigma**2/(lr*rBar**3)))/8 + (5*(sigma**2/(lr**2*rBar**2)))/16 + (507*(sigma**3/rBar**4))/8 + \
		(59*(sigma**3/(lr*rBar**3)))/8 + (537*(sigma**4/rBar**4))/16)
	]

deriv1DenomLambdasCfuns = [ \
		"#define denom(xi,beta,sigma,lr,rBar) (-(pow(rBar, -2)*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi)) - (sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*pow(rBar, -2)*sin(xi/2))/2 + (sigma*cos(beta)*pow(lr, -1)*pow(rBar, -1)*sin(xi/2))/2)"
	]


