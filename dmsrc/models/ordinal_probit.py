import sys
import time
import numpy as np
import scipy as sp
from scipy import linalg
from scipy import stats
from ._ordinal_probit import _sample_ordinal_probit

class OrdinalProbit():
	"""
	Ordinal probit model for modeling debate results.

	Parameters
	----------
	X : np.array of ints
		(k, n, 2)-shaped array of speaker ids.
		n = number of rounds in the dataset
		k = number of teams per round (4 in BP)
		X[l, i, j] = speaker ID of speaker j on team l
		in round i.
	ranks : np.array of ints
		(k, n)-shaped array of results.
		ranks[l, i] = num. pts that team l earned in round i.
		In outrounds, advancing = 3 pts, elim = 0 pts.
	sigma2_a0 : float
		`sigma2`` has an InvGamma(``sigma2_a0``, ``sigma2_b0``) hyperprior.
	sigma2_b0 : float
		`sigma2`` has an InvGamma(``sigma2_a0``, ``sigma2_b0``) hyperprior.

	tau2_a0 : float
		`tau2`` has an InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior.
	tau2_b0 : float
		`tau2`` has an InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior.
	"""

	def __init__(
		self, 
		X, 
		ranks,
		sigma2_a0=2,
		sigma2_b0=1,
		tau2=1,
		tau2_a0=2,
		tau2_b0=1,
		update_tau2=True,
	):
		# save input data
		self.k, self.n, _ = X.shape
		self.X = X.astype(int)
		if not self.X.flags['C_CONTIGUOUS']:
			self.X = np.ascontiguousarray(self.X)
		self.ranks = ranks.astype(int)
		if not self.ranks.flags['C_CONTIGUOUS']:
			self.ranks = np.ascontiguousarray(self.ranks)
		# precompute l2 norm of design matrix
		# (we use a sparse representation of the design)
		self.p = np.max(self.X) + 1 # total number of speaker IDs

		# save hyperparams
		self.sigma2_a0 = sigma2_a0
		self.sigma2_b0 = sigma2_b0
		self.tau2_a0 = tau2_a0
		self.tau2_b0 = tau2_b0
		self.tau2 = tau2
		self.update_tau2 = update_tau2

	def sample(
		self,
		N,
		burn=None,
		chains=1,
		log_interval=0,
	):
		"""
		Parameters
		----------
		N : int
			Number of samples per chain
		burn : int
			Number of samples to burn per chain. Defaults to 0.1 N.
		chains : int
			Number of chains to run. Defaults to 1.
		log_interval : int
			Reports progress every log_interval iterations.
			If log_interval=0, does not report at all.
		"""
		if burn is None:
			burn = int(0.1 * N)

		out = []
		for _ in range(chains):
			out.append(_sample_ordinal_probit(
				N=N+burn,
				X=self.X,
				ranks=self.ranks,
				tau2=self.tau2,
				update_tau2=self.update_tau2,
				sigma2_a0=self.sigma2_a0,
				sigma2_b0=self.sigma2_b0,
				tau2_a0=self.tau2_a0,
				tau2_b0=self.tau2_b0,
				log_interval=log_interval,
			))

		# concatenate
		self.betas = np.concatenate([x['betas'][burn:] for x in out])
		self.mus = np.concatenate([x['mus'][burn:] for x in out])
		self.tau2s = np.concatenate([x['tau2s'][burn:] for x in out])
		self.sigma2s = np.concatenate([x['sigma2s'][burn:] for x in out])