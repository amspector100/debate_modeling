import time
import numpy as np
import scipy as sp
from scipy import stats
import unittest
import pytest
import os
import sys
try:
	from . import context
	from .context import dmsrc
	import dmsrc.models._truncnorm as tn
# For profiling
except ImportError:
	import context
	from context import dmsrc
	import dmsrc.models._truncnorm as tn

class TestTruncNorm(unittest.TestCase):
	"""
	tests
	"""
	def test_truncnorm(self):
		reps = 100000
		q = 1 / 10000 # nominal level
		intervals = [
			(-10, 10),
			(-0.5, 0.5),
			(0, 2),
			(1, 1.01),
			(0.5, 3),
			(1, 4),
		]
		mus = [0, 5]
		sigma2s = [1, 100]
		for alpha, beta in intervals:
			for a, b in zip([alpha, -beta], [beta, -alpha]):
				for mu, sigma2 in zip(mus, sigma2s):
					msg = f"at a={a}, b={b}, mu={mu}, sigma2={sigma2}."
					print(msg)
					scipy_rvs = stats.truncnorm(
						loc=mu,
						scale=np.sqrt(sigma2),
						a=(a - mu) / np.sqrt(sigma2),
						b=(b - mu) / np.sqrt(sigma2),
					).rvs(size=reps)
					tnrvs = np.array([
						tn.sample_truncnorm(
							mean=mu, var=sigma2, a=a, b=b
						)
						for _ in range(reps)
					])
					# test range restrictions
					minval, maxval = tnrvs.min(), tnrvs.max()
					self.assertTrue(
						minval >= a, f"min sampled val = {minval} {msg}"
					)
					self.assertTrue(
						maxval <= b, f"max sampled val = {maxval} {msg}"
					)
					# Test equality of means
					diffs = scipy_rvs - tnrvs
					sp_mean, tn_mean = scipy_rvs.mean(), tnrvs.mean()
					tstat = np.sqrt(reps) * diffs.mean() / diffs.std()
					pval = 2 * (1 - stats.norm.cdf(np.abs(tstat)))
					self.assertTrue(
						pval >= q, f"pval={pval}, scipy mean={sp_mean}, new mean = {tn_mean} for reps={reps} {msg}"
					)

if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestEx(), TestEx2()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()