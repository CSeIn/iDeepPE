## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.
from scipy.special import exp1, i0, i1
import torch.special as special
import math
import torch
import torch.nn as nn
import numpy as np

def mmse_stsa(xi, gamma):
	"""
	Computes the MMSE-STSA gain function.

	Numpy version:
		nu = np.multiply(xi, np.divide(gamma, np.add(1, xi)))
		G = np.multiply(np.multiply(np.multiply(np.divide(np.sqrt(np.pi), 2),
			np.divide(np.sqrt(nu), gamma)), np.exp(np.divide(-nu,2))),
			np.add(np.multiply(np.add(1, nu), i0(np.divide(nu,2))),
			np.multiply(nu, i1(np.divide(nu, 2))))) # MMSE-STSA gain function.
		idx = np.isnan(G) | np.isinf(G) # replace by Wiener gain.
		G[idx] = np.divide(xi[idx], np.add(1, xi[idx])) # Wiener gain.
		return G

	Argument/s:
		xi - a priori SNR.
		gamma - a posteriori SNR.

	Returns:
		G - MMSE-STSA gain function.
	"""
	pi = torch.tensor(math.pi)
	xi = torch.clamp(xi, min=1e-12) 
	gamma = torch.clamp(gamma, min=1e-12)
	nu = torch.mul(xi, torch.div(gamma, torch.add(1.0, xi)))
	G = torch.mul(torch.mul(torch.mul(torch.div(torch.sqrt(pi), 2.0),
		torch.div(torch.sqrt(nu), gamma)), torch.exp(torch.div(-nu, 2.0))),
		torch.add(torch.mul(torch.add(1.0, nu), torch.i0(torch.div(nu, 2.0))),
		torch.mul(nu, torch.special.i1(torch.div(nu, 2.0))))) # MMSE-STSA gain function.
	G_WF = wf(xi)
	logical = tf.math.logical_or(tf.math.is_nan(G), tf.math.is_inf(G))
	G = tf.where(logical, G_WF, G)
	return G

def mmse_lsa(xi, gamma):
	"""
	Computes the MMSE-LSA gain function.

	Numpy version:
		v_1 = np.divide(xi, np.add(1.0, xi))
		nu = np.multiply(v_1, gamma)
		return np.multiply(v_1, np.exp(np.multiply(0.5, exp1(nu)))) # MMSE-LSA gain function.

	Argument/s:
		xi - a priori SNR.
		gamma - a posteriori SNR.

	Returns:
		MMSE-LSA gain function.
	"""
	xi = torch.clamp(xi, min=1e-12) 
	gamma = torch.clamp(gamma, min=1e-12)
	v_1 =torch.div(xi, 1+xi)
	
	nu = torch.mul(v_1, gamma).cpu().detach().numpy()
	v_2 = torch.tensor(exp1(nu)).cuda()

	# v = torch.mul(v_1, gamma)
	# v_2 = special.expit(v)
	## v_2 = tf.math.negative(tf.math.special.expint(tf.math.negative(nu))) # E_1(x) = -E_i(-x)
	return torch.mul(v_1, torch.exp(0.5 * v_2)) # MMSE-LSA gain function.

def wf(xi):
	"""
	Computes the Wiener filter (WF) gain function.

	Argument/s:
		xi - a priori SNR.

	Returns:
		WF gain function.
	"""
	return torch.div(xi, torch.add(xi, 1.0)) # WF gain function.

def srwf(xi):
	"""
	Computes the square-root Wiener filter (WF) gain function.

	Argument/s:
		xi - a priori SNR.

	Returns:
		SRWF gain function.
	"""
	return torch.sqrt(wf(xi)) # SRWF gain function.

def cwf(xi):
	"""
	Computes the constrained Wiener filter (WF) gain function.

	Argument/s:
		xi - a priori SNR.

	Returns:
		cWF gain function.
	"""
	return wf(torch.sqrt(xi)) # cWF gain function.

def dgwf(xi, cdm):
	"""
	Computes the dual-gain Wiener filter (WF).

	Argument/s:
		xi - a priori SNR.
		cdm - constructive-deconstructive mask.

	Returns:
		G - DGWF.
	"""
	v_1 = torch.divide(2.0, torch.tensor(math.pi))
	v_2 = torch.mul(2, v_1)
	v_3 = torch.sqrt(xi)
	v_4 = torch.add(xi, 1.0)
	G_minus = torch.divide(torch.subtract(xi, torch.mul(v_1, v_3)),
		torch.subtract(v_4, torch.mul(v_2, v_3)))
	G_plus = torch.divide(torch.add(xi, torch.mul(v_1, v_3)),
		torch.add(v_4, torch.mul(v_2, v_3)))
	G = torch.where(cdm, G_plus, G_minus)
	return G # DGWF.

def irm(xi):
	"""
	Computes the ideal ratio mask (IRM).

	Argument/s:
		xi - a priori SNR.

	Returns:
		IRM.
	"""
	return srwf(xi) # IRM.

def ibm(xi):
	"""
	Computes the ideal binary mask (IBM) with a threshold of 0 dB.

	Argument/s:
		xi - a priori SNR.

	Returns:
		IBM.
	"""
	return torch.greater(xi, 1.0).type(torch.float32) # IBM (1 corresponds to 0 dB).



def deepmmse(xi, gamma):
    return torch.pow( torch.div(1, 1+xi), 2) + torch.div(xi, torch.mul(gamma, 1+xi))# MMSE noise periodogram estimate gain function.

def gfunc(xi, gamma=None, gtype=None, cdm=None):
	"""
	Computes the selected gain function.

	Argument/s:
		xi - a priori SNR.
		gamma - a posteriori SNR.
		gtype - gain function type.
		cdm - constructive-deconstructive mask.

	Returns:
		G - gain function.
	"""
	if gtype == 'mmse-lsa': G = mmse_lsa(xi, gamma)
	elif gtype == 'mmse-stsa':  G = mmse_stsa(xi, gamma)
	elif gtype == 'wf': G = wf(xi)
	elif gtype == 'srwf': G = srwf(xi)
	elif gtype == 'cwf': G = cwf(xi)
	elif gtype == 'dgwf': G = dgwf(xi, cdm)
	elif gtype == 'irm': G = irm(xi)
	elif gtype == 'ibm': G = ibm(xi)
	elif gtype == 'deepmmse': G = deepmmse(xi, gamma)
	else: raise ValueError('Invalid gain function type.')
	return G
