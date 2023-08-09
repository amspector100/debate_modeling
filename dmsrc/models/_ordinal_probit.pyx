# cython: profile=True

import sys
import time 
cimport cython
import numpy as np
import scipy.stats
cimport numpy as np
from numpy cimport PyArray_ZEROS
import scipy.linalg
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp, fabs, sqrt, fmin, fmax, erfc

# Fast uniform/truncated normal sampling
from ._truncnorm import random_uniform, sample_truncnorm

# Blas commonly used parameters
cdef double zero = 0, one = 1, neg1 = -1
cdef int inc_0 = 0;
cdef int inc_1 = 1
cdef char* trans_t = 'T'
cdef char* trans_n = 'N'
cdef char* triang_u = 'U'
cdef char* triang_l = 'L'
cdef double M_SQRT1_2 = sqrt(0.5)

# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
def _sample_ordinal_probit(
	int N,
	long[:, :, ::1] X, # k x n x 2
	long[:, ::1] ranks, # k x n
	double tau2,
	int update_tau2, 
	double tau2_a0,
	double tau2_b0,
	double sigma2_a0,
	double sigma2_b0,
	int log_interval,

):
	# Initialize outputs
	cdef:
		# Useful constants
		int k = X.shape[0]
		int n = X.shape[1]
		int nk = n * k
		int p = np.max(X) + 1
		int i, it, j, ii, jj, kk, kk2
		int sid0, sid1 # speaker ids

		# Initialize outputs
		np.ndarray[long, ndim=1] inds = np.arange(p)
		np.ndarray[double, ndim=2] betas_arr = np.zeros((N, p))
		double[:, ::1] betas = betas_arr
		np.ndarray[double, ndim=1] sigma2s_arr = np.zeros(N,)
		double[::1] sigma2s = sigma2s_arr
		np.ndarray[double, ndim=1] tau2s_arr = np.zeros(N,)
		double[::1] tau2s = tau2s_arr
		# latent variables
		np.ndarray[double, ndim=3] mus_arr = np.zeros((N, k, n))
		double[:, :, ::1] mus = mus_arr

		# Precompute useful quantities 
		double[::1] logdets = np.zeros((p,))
		double[::1] post_vars = np.zeros((p,))
		double[::1] Xl2 = np.zeros((p, ))

		# Scratch
		double ranks_mu, ranks_var, ranks_sd
		double mumean, lbound, ubound, XjTr, old_betaj, muresid
		double postmean, r2, sigma_b
		double sigma, tau
		# residuals and predictions
		# np.ndarray[double, ndim=1] r_arr = np.zeros((k, n))
		# double[:, ::1] r = r_arr
		# np.ndarray[double, ndim=1] preds_arr = np.zeros((k, n))
		# double[:, ::1] preds = preds_arr

	# precompute l2 norm of dense design matrix
	# (although X is a sparse representation)
	for kk in range(k):
		for ii in range(n):
			sid0 = X[kk, ii, 0]
			sid1 = X[kk, ii, 1]
			if sid0 == sid1:
				Xl2[sid0] += 4
			else:
				Xl2[sid0] += 1
				Xl2[sid1] += 1

	# precompute inverse gamma r.v.s for resampling tau,sigma2
	cdef double sigma_a = (n * k) / 2.0 + sigma2_a0
	cdef double tau_a = (n * k) / 2.0 + tau2_a0
	cdef np.ndarray [double, ndim=1] invgam_sigma = scipy.stats.invgamma(sigma_a).rvs(N)
	cdef np.ndarray [double, ndim=1] invgam_tau = scipy.stats.invgamma(tau_a).rvs(N)

	# initialize
	sigma2s[0] = 1.0
	tau2s[0] = tau2

	## initialize mus
	# start by finding normalizations on rank
	ranks_mu = 0; ranks_var = 0
	for kk in range(k):
		for ii in range(n):
			ranks_mu += ranks[kk, ii]
			ranks_var += ranks[kk, ii]**2
	ranks_mu = ranks_mu / (n * k)
	ranks_var = ranks_var / (n * k) - (ranks_mu**2)
	ranks_sd = sqrt(ranks_var)
	# now initialize mus
	for kk in range(k):
		for ii in range(n):
			mus[0, kk, ii] = (ranks[kk, ii] - ranks_mu) / ranks_sd


	# loop through and sample
	for i in range(N):
		sigma = sqrt(sigma2s[i])
		tau = sqrt(tau2s[i])
		if log_interval != 0:
			if i % log_interval == 0:
				print(f"Beginning iteration={i}.")
				sys.stdout.flush()
		# precompute log determinants / posterior variances
		for j in range(p):
			logdets[j] = log(1.0 + tau2s[i] * Xl2[j] / sigma2s[i]) / 2.0
			post_vars[j] = 1.0 / (1.0 / tau2s[i] + Xl2[j] / sigma2s[i])
		## udpate mu
		## TODO: this is the bottleneck and does not use all cores.
		## try https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html
		## to do this, need to switch the order of the loops to be 
		## outer = {for ii in range(n)}, inner = {for kk in range(k)} 
		for kk in range(k):
			for ii in range(n):
				# mean of mu
				mumean = betas[i, X[kk, ii, 0]] + betas[i, X[kk, ii, 1]]
				# compute truncation based on other mus and ranks
				lbound = - 100 * sigma
				ubound = 100 * sigma
				for kk2 in range(k):
					if ranks[kk, ii] == ranks[kk2, ii]:
						continue
					elif ranks[kk, ii] < ranks[kk2, ii]:
						ubound = fmin(ubound, mus[i, kk2, ii]) 
					else:
						lbound = fmax(lbound, mus[i, kk2, ii])
				mus[i, kk, ii] = sample_truncnorm(
					mean=mumean,
					sd=sigma,
					a=lbound,
					b=ubound,
				)

		## update beta
		np.random.shuffle(inds)
		for j in inds:
			old_betaj = betas[i, j]
			betas[i, j] = 0
			# compute XjTr leveraging sparsity of X
			XjTr = 0
			for kk in range(k):
				for ii in range(n):
					sid0 = X[kk, ii, 0]; sid1 = X[kk, ii, 1]
					if sid0 == j or sid1 == j:
						mumean = betas[i, sid0] + betas[i, sid1]
						muresid = mus[i, kk, ii] - mumean
					if sid0 == j:
						XjTr += muresid
					if sid1 == j:
						XjTr += muresid
			post_mean = post_vars[j] * XjTr / sigma2s[i]
			betas[i, j] = np.sqrt(post_vars[j]) * np.random.randn() + post_mean

		# Update hyperparams
		# 1. tau2
		if update_tau2 == 1:
			r2 = blas.dnrm2(&p, &betas[i, 0], &inc_1)
			tau_b = r2 * r2 / 2.0 + tau2_b0
			tau2s[i] = tau_b * invgam_tau[i]

		# 2. sigma2: currently unidentifiable
		sigma2s[i] = 1.0
		# r2 = blas.dnrm2(&nk, &mus[i, 0, 0], &inc_1)
		# sigma_b = r2 * r2 / 2.0 + sigma2_b0
		# sigma2s[i] = sigma_b * invgam_sigma[i]

		# Set new betas, p0s to be old values (temporarily)
		if i != N - 1:
			betas[i+1] = betas[i]
			mus[i+1] = mus[i]
			sigma2s[i+1] = sigma2s[i]
			tau2s[i+1] = tau2s[i]

	return {"mus":mus_arr, "betas":betas_arr, "tau2s":tau2s_arr, "sigma2s":sigma2s_arr}