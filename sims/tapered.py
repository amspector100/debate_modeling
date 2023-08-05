"""
Template for running simulations.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from context import dmsrc
from dmsrc import parser, utilities
import dmsrc.tournsim as ts

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

# Holds all of the pairing and weighting functions
PAIR_FNS = {'random':ts.random_pair, 'power':ts.power_pair}
WEIGHT_FNS = {
	'constant':ts.constant_weighting,
	'barnes':ts.barnes_weighting
}

def single_seed_sim(
	**args
):
	# Create skills
	nteam = args.get("nteam")
	nbreak = args.get("nbreak")
	nrounds = args.get("nrounds")
	np.random.seed(args.get("skill_seed", 1))
	skills = 77 + 2 * np.random.randn(nteam)

	# Run tournament
	np.random.seed(args.get("seed"))
	results = ts.run_tournament(
		skills=skills, 
		pair_fn=PAIR_FNS[args.get('pairing')],
		nrounds=nrounds,
		weights=WEIGHT_FNS[args.get("weight")](nrounds),
	)

	# Save data
	raw_output = ts.result_to_be_saved(results, param_dict=args)
	params = sorted(list(args.keys()))
	summary = pd.Series(
		[args[x] for x in params], index=params
	)	
	for metric, name in zip(
		[ts.avg_break_skill, ts.max_nonbreaking_skill], 
		['avg_break_skill', 'max_nonbreaking_skill']
	):
		summary[name] = metric(results, nbreak=nbreak)
	return raw_output, summary

def main(args):
	# Parse arguments
	args = parser.parse_args(args)
	reps = args.pop('reps', [1])[0]
	num_processes = args.pop('num_processes', [1])[0]

	# Key defaults go here
	args['nteam'] = args.get("nteam", [100])
	args['nbreak'] = args.get("nbreak", [16])
	args['pairing'] = args.get("pairing", ['power'])
	args['nrounds'] = args.get("nrounds", [5])
	args['weight'] = args.get("weight", ['constant'])

	# Save args, create output dir
	output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
	args.pop("description")

	# Run outputs
	outputs = utilities.apply_pool_factorial(
		func=single_seed_sim,
		seed=list(range(1, reps+1)), 
		num_processes=num_processes, 
		**args,
	)
	raw_outputs = pd.concat([x[0] for x in outputs], axis='index')
	summaries = pd.DataFrame([x[1] for x in outputs])
	raw_outputs.to_csv(output_dir + "raw.csv", index=False)
	summaries.to_csv(output_dir + "summary.csv", index=False)

if __name__ == '__main__':
	main(sys.argv)