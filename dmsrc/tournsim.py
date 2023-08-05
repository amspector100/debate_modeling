import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

###############################################
# Functions for evaluating tournament results #
###############################################

def avg_break_skill(results, nbreak=32):
    results = results.sort_values('points', ascending=False)
    return results.iloc[0:nbreak]['skill'].mean()

def max_nonbreaking_skill(results, nbreak=32):
    results = results.sort_values('points', ascending=False)
    return results.iloc[nbreak]['skill'].max()

###############################################
########### Weighting functions ###############
###############################################

def constant_weighting(nrounds):
	return np.ones(nrounds)

def barnes_weighting(nrounds):
	weights = np.ones(nrounds)
	n2 = int(nrounds / 2)
	weights[0:n2] = 2.0
	weights[0] = 3.0
	return weights

###############################################
# Functions for running simulated tournaments #
###############################################

def power_pair(points):
	""" power pairing with random tiebreaks """
	nteam = points.shape[0]
	rooms = np.zeros(nteam)
	pts_rand = points + np.random.uniform(0, 1, size=nteam)
	inds = np.argsort(pts_rand)
	rooms[inds] = np.arange(nteam) // 4
	return rooms

def random_pair(points):
	""" random pairing """
	nteam = points.shape[0]
	rooms = np.arange(nteam) // 4
	np.random.shuffle(rooms)
	return rooms

def run_tournament(skills, pair_fn, nrounds=5, weights=None):
	"""
	Parameters
	----------
	skills : np.array
		nteam length array of skills
	pair_fn : np.array
		function which takes in points array and returns list of rooms
	weights : np.array
		nrounds length array of how to weight each round
	"""
	# weighting by points for tournament
	if weights is None:
		weights = np.ones(nrounds)
	# number of teams
	nteam = skills.shape[0]
	results = pd.DataFrame(
		skills, index=np.arange(nteam), columns=['skill']
	)
	results['points'] = 0.0
	for rnum in range(nrounds):
		results[f'room{rnum}'] = pair_fn(results['points'])
		results[f'skill{rnum}'] = results['skill'] + 2 * np.random.randn(nteam)
		results[f'result{rnum}'] = results.groupby(f'room{rnum}')[f'skill{rnum}'].rank()
		results['points'] += weights[rnum] * (results[f'result{rnum}'] - 1)
	return results

###############################################
####### Clean final tournament results ########
###############################################

def result_to_be_saved(results, param_dict=None):
	# parse which columns to save
	cols = ['skill', 'points']
	for c in results.columns:
		if 'result' in c:
			cols.append(c)
	# Subset to only these columns
	sub = results[cols].copy()
	# Add parameters
	if param_dict is not None:
		for key in param_dict:
			sub[key] = param_dict[key]
	return sub