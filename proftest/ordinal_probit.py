import time
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import unittest
import pytest
import os
import sys
try:
	from . import context
	from .context import dmsrc
	from dmsrc.models.ordinal_probit import OrdinalProbit
# For profiling
except ImportError:
	import context
	from context import dmsrc
	from dmsrc.models.ordinal_probit import OrdinalProbit

class TestOrdinalProbit(unittest.TestCase):
	"""
	tests
	"""
	def test_ordinal_probit(self):
		# Load data
		rdf = pd.read_csv("data/combined/round_data.csv")
		n = len(rdf)
		p = rdf[[c for c in rdf.columns if 'speaker_id' in c]].max().max() + 1
		k = 4
		X = np.zeros((k, n, 2))
		ranks = np.zeros((k, n))
		for kk, side in zip(np.arange(k), ['OG', 'OO', 'CG', 'CO']):
			ranks[kk] = rdf[side+"_rank"]
			X[kk,:,0] = rdf[side+"_speaker_id0"]
			X[kk,:,1] = rdf[side+"_speaker_id1"]

		# fit model
		ordprob = OrdinalProbit(
		    X=X, ranks=ranks
		)
		ordprob.sample(N=100, log_interval=1)

if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'proftest/{basename}':
		time0 = time.time()
		context.run_all_tests([TestOrdinalProbit()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()