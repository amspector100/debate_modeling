# cython: profile=False

######################################################################
# Custom (faster) truncated normal sampler.
# See http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.26.6892
# Follows implementation in R:
# https://github.com/olafmersmann/truncnorm/blob/master/src/rtruncnorm.c
######################################################################
cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp, fabs, sqrt, fmin, fmax, M_1_PI
#from numpy.math cimport INFINITY

cdef double INV_SQRT2PI = sqrt(M_1_PI / 2)


@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double random_uniform():
	cdef double r = rand()
	return r / RAND_MAX

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _expo_tn_sampler(double a, double b):
	"""
	Samples Z \sim N(0,1) | Z \in [a,b]
	using an exponential rejection sampler.
	"""
	cdef double rho
	cdef double u 
	cdef double y # ~ expo(a)
	while True:
		y = random_uniform()
		y = -1*log(y) / a
		rho = exp(-0.5*y*y)
		u = random_uniform()
		if u <= rho and y + a <= b:
			return y + a

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _norm_tn_sampler(double a, double b):
	"""
	Samples Z \sim N(0,1) | Z \in [a,b]
	using a gaussian rejection sampler.
	"""
	cdef double z
	while True:
		z = np.random.randn()
		if a >= 0:
			z = fabs(z)
		if b <= 0:
			z = - fabs(z)
		if z >= a and z <= b:
			return z

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _unif_tn_sampler(double a, double b):
	"""
	Samples Z \sim N(0,1) | Z \in [a,b]
	using a uniform(a, b) rejection sampler.
	"""
	cdef double z, u, ub, c
	cdef double diff = b - a
	# upper bound on normal density on [a,b]
	# case 1: if 0 in interval, density maximized at 0
	if a <= 0 and b >= 0:
		ub = 1
	else:
		c = fmin(fabs(a), fabs(b))
		ub = exp(-c*c/2)

	while True:
		# sample uniform
		z = a + diff * random_uniform()
		# rejection sample
		u = ub * random_uniform()
		if u <= exp(-z*z/2):
			return z

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _sample_truncnorm_std(double a, double b, double phia, double phib):
	"""
	Samples Z \sim N(0,1) | Z \in [a, b]
	efficiently.
	"""

	# constants from the paper
	cdef double t1 = 0.150
	cdef double t2 = 2.18
	cdef double t3 = 0.725
	cdef double t4 = 0.45
	cdef double ratio
	## Case 1
	if a <= 0 and b >= 0:
		if phia <= t1 or phib <= t1:
			return _norm_tn_sampler(a, b)
		else:
			return _unif_tn_sampler(a, b)
	## Case 2
	if a > 0:
		ratio = phia / phib
		if ratio <= t2:
			return _unif_tn_sampler(a, b)
		else:
			if a > t3:
				return _norm_tn_sampler(a, b)
			else:
				return _expo_tn_sampler(a, b)
	# else, exploit symmetry		
	return -1 * _sample_truncnorm_std(-b, -a, phib, phia)


@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double sample_truncnorm(
	double mean,
	double var,
	double a,
	double b,
):
	"""
	samples Z ~ N(mean, var) | Z in [a,b]
	"""
	if a >= b:
		raise ValueError("a >= b")
	cdef double scale = sqrt(var)
	# adjust
	a = (a - mean) / scale
	b = (b - mean) / scale
	cdef double phia = exp(-a*a/2) * INV_SQRT2PI
	cdef double phib = exp(-b*b/2) * INV_SQRT2PI
	# sample
	z = _sample_truncnorm_std(a, b, phia, phib)
	# return
	return mean + scale * z