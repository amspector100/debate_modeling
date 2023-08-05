import time
import numpy as np
import pandas as pd
import glob
import copy
from collections import Counter
from unidecode import unidecode

### NOTE ###
# any team containing the word "swing"
# is automatically assigned a dummy speaker
### END NOTE ###
all_speakers_fname = "../data/combined/all_speakers.csv"

def not_nan(x):
	if not isinstance(x, float):
		return True
	else:
		return not np.isnan(x)

def is_float(x):
	try:
		float(x)
		return True
	except:
		return False

def list_series_to_array(series):
	return np.array([np.array(x) for x in series])

def clean_speaker_df(sdf, all_speakers=None):
	# Correctly process names: replace two periods with one
	sdf = sdf.loc[sdf['name'].notnull()]
	si = sdf.index
	sdf['name'] = sdf['name'].str.lower()
	sdf['name'] = sdf['name'].str.replace("  ", " ")
	sdf['name'] = sdf['name'].apply(unidecode)

	# all swing/missing speakers are aggregated
	swing_flags = sdf['team'].str.lower().str.contains("swing")
	swing_flags = sdf['name'].str.contains("speaker")
	sdf.loc[swing_flags, 'name'] = 'swing_speaker'

	# Add ids
	if all_speakers is not None:
		sdf['id'] = sdf['name'].map(
			all_speakers.set_index("name")['id']
		)
	return sdf

def create_global_speaker_list(tournaments):
	"""
	To do: 
	1. Combine similar names that are different spellings
	"""

	# speaker data frame
	codes = tournaments['Code']
	#columns = ['speaker'] + codes
	sdfs = dict()

	# Collect all names
	names = []
	for code in codes:
		# load and clean
		tourndir = f"../data/{code}/raw/"
		sdf = pd.read_csv(tourndir + "speakers.csv", sep='\t')
		sdf = clean_speaker_df(sdf)

		dup_flags = sdf.duplicated("name", keep=False)
		dup_flags = dup_flags & (sdf['name'] != "swing_speaker")
		if np.any(dup_flags):
			print(f"\nAt {code}, some duplicated speaker names:\n{sdf.loc[dup_flags]}")
			print("By default, treating all of these as the same person.")
			sdf = sdf.loc[~sdf.duplicated("name", keep='first')]

		# Add names
		names += sdf['name'].tolist()
		sdf.set_index("name", inplace=True)
		sdfs[code] = sdf

	# Create output
	names = set([x for x in names if not_nan(x)])
	all_speakers = pd.DataFrame(
		data=np.nan, index=sorted(list(names)), columns=['id']
	)
	all_speakers.index.name = 'name'
	all_speakers['id'] = np.arange(all_speakers.shape[0])
	# # for future compatability
	# cleaned_names = set(all_speakers.index.tolist())
	# for code in codes:
	# 	sdf = sdfs[code]
	# 	in_tourny = set(sdf.index.tolist()).intersection(cleaned_names)
	# 	in_tourny = sorted(list(in_tourny))
	# 	all_speakers.loc[in_tourny, code] = sdf.loc[in_tourny, "team"].values

	all_speakers.to_csv(all_speakers_fname)

def process_single_round(rdf, team2speakerid, swing_id):
	# Missing teams get mapped to the swing speaker
	rteams = set(rdf['team'].unique().tolist())
	steams = set(team2speakerid.index.tolist())
	diff = rteams - steams
	if len(diff) > 0:
		for d in diff:
			team2speakerid.loc[d] = [swing_id]

	# Find speaker ids
	sids = team2speakerid.loc[rdf['team']]
	sids = sids.apply(lambda x: [x[0], x[0]] if len(x) == 1 else x)
	sids = sids.apply(lambda x: x[0:2] if len(x) > 2 else x)
	rdf['sids'] = sids.values

	# infer the type of round
	if rdf.shape[0] == 4:
		return process_finals(rdf)
	elif '1st' in rdf['result'].unique():
		return process_single_inround(rdf)
	elif 'advancing' in rdf['result'].unique():
		return process_single_outround(rdf)
	else:
		raise ValueError("Unsure if round is inround, outround, or finals")

def process_finals(rdf):
	winflag = rdf['result'].isin(['1st', 'advancing'])
	winners = list_series_to_array(rdf.loc[winflag, 'sids'])
	winners = winners.reshape(1, 2)
	participants = np.stack(
		[np.array(x).reshape(1, 2) for x in rdf['sids'].tolist()],
		axis=-1
	)
	return [winners], [participants]

def process_single_outround(rdf):
	rdf = rdf[['sids', 'adjudicators', 'result']]
	rdf = rdf.sort_values(['adjudicators'])
	if rdf.shape[0] % 4 != 0:
		raise ValueError(f"N. teams in outround is not divisible by 4:\n {rdf}")
	# label the first advancer and loser 
	# (this is purely for indexing purposes)
	firsts = np.zeros(rdf.shape[0]).astype(bool)
	for j in range(rdf.shape[0]):
		# reset for each round
		if j % 4 == 0:
			adv_first = True
			elim_first = True
		row = rdf.iloc[j]
		if row['result'] == 'advancing':
			if adv_first:
				firsts[j] = True
				adv_first = False
		else:
			if elim_first:
				firsts[j] = True
				elim_first = False
	rdf['_first'] = firsts

	all_winners = []
	all_participants = []
	adv_flags = rdf['result'] == 'advancing'
	for adv_first in [True, False]:
		all_winners.append(list_series_to_array(
			rdf.loc[(adv_flags) & (rdf['_first'] == adv_first), 'sids']
		))
		participants = [all_winners[-1]]
		for elim_first in [True, False]:
			participants.append(list_series_to_array(
				rdf.loc[(~adv_flags) & (rdf['_first'] == elim_first), 'sids']
			))
		participants = np.stack(participants, axis=-1)
		all_participants.append(participants)

	return all_winners, all_participants


def process_single_inround(rdf):
	"""
	rfd must be preprocessed
	"""
	# Ensure there is exactly one 1st, 2nd, 3rd, 4th per round
	counts = rdf.groupby(["adjudicators", "result"])['team'].count().reset_index()
	bad_rds = counts.loc[counts['team'] != 1, 'adjudicators'].unique()
	rdf = rdf.loc[~rdf['adjudicators'].isin(bad_rds)]

	# Sort by round
	rdf = rdf.sort_values(["adjudicators", "result"]).reset_index()
	# For each team in each round, find the speakers associated 
	# and separate by rank. Note these are in the same order
	speakers_by_rank = {}
	for rank, rstr in enumerate(["1st", "2nd", "3rd", "4th"]):
		sids = rdf.loc[rdf['result'] == rstr, 'sids']
		speakers_by_rank[rank] = list_series_to_array(sids)

	# Turn into index arrays that can be used in the cvxpy call
	all_winners = []
	all_participants = []
	for rank in range(3):
		all_winners.append(speakers_by_rank[rank])
		participants = []
		for lower_rank in range(rank, 4):
			participants.append(speakers_by_rank[lower_rank])
		#participants = np.concatenate(participants, axis=-1)
		all_participants.append(np.stack(participants, axis=-1))

	return all_winners, all_participants

def process_rounds_data(tournaments, all_speakers):
	"""
	Assumes that all_speakers has already been created
	"""
	swing_id = all_speakers.loc[
		all_speakers['name'].str.contains('speaker'), 'id'
	].item()
	codes = tournaments['Code']

	# Initialize data output
	all_winners = []
	all_participants = []
	for code in codes:
		print(f"At code={code}.")
		# Load speaker data
		tourndir = f"../data/{code}/raw/"
		sdf = pd.read_csv(tourndir + "speakers.csv", sep='\t')
		sdf = clean_speaker_df(sdf, all_speakers=all_speakers)
		# Map teams to speakers
		team2speakerid = sdf.groupby("team")['id'].apply(
			lambda x: list(np.unique(x))
		)
		# Load round data
		rfiles = sorted(glob.glob(tourndir + "round*.csv"))
		for rfile in rfiles:
			# check this is really round results
			round_number = rfile.split("round")[-1].split(".csv")[0]
			if is_float(round_number):
				rdf = pd.read_csv(rfile, sep='\t')
				winners, participants = process_single_round(
					rdf=rdf, team2speakerid=team2speakerid, swing_id=swing_id
				)
				all_winners.extend(winners)
				all_participants.extend(participants)

	# Concatenate together into three large arrays
	dims = [2, 3, 4]
	winner_dict = {k:[] for k in dims}
	participant_dict = {k:[] for k in dims}
	for winners, participants in zip(all_winners, all_participants):
		d = participants.shape[-1]
		winner_dict[d].append(winners)
		participant_dict[d].append(participants)
	for d in dims:
		winner_dict[d] = np.concatenate(winner_dict[d], axis=0)
		np.save(f"../data/combined/winners{d}.npy", winner_dict[d])
		participant_dict[d] = np.concatenate(participant_dict[d], axis=0)
		np.save(f"../data/combined/participants{d}.npy", participant_dict[d])

def process_speaker_data(tournaments, all_speakers):
	# swing_id = all_speakers.loc[
	# 	all_speakers['name'].str.contains('speaker'), 'id'
	# ].item()
	# Initialize data output
	all_winners = []
	all_participants = []
	sdfs = []
	for cid, code in enumerate(tournaments['Code']):
		# Load speaker data
		tourndir = f"../data/{code}/raw/"
		sdf = pd.read_csv(tourndir + "speakers.csv", sep='\t')
		sdf = clean_speaker_df(sdf, all_speakers=all_speakers)
		sdf['tourn'] = code
		sdf['tourn_id'] = cid
		sdfs.append(sdf)

	# Combine
	sdf = pd.concat(sdfs, axis='index')
	sdf[sdf == '—'] = np.nan
	# infer which columns are speaks for rounds
	ids = ['id', 'tourn', 'tourn_id']
	rel_cols = copy.copy(ids)
	for c in sdf.columns:
		cl = c.lower()
		if cl[0] == 'r' and is_float(cl.split('r')[-1]):
			rel_cols.append(c)
	sdf = sdf[rel_cols].melt(id_vars=ids)[ids + ['value']]
	sdf = sdf.loc[sdf['value'].notnull()]
	sdf.to_csv("../data/combined/all_speaks.csv", index=False)

def main():
	t0 = time.time()

	# Read list of tournaments
	tournaments = pd.read_csv("../data/tournaments.csv")

	# Create global speaker list
	create_global_speaker_list(tournaments)

	# Read list of speakers
	all_speakers = pd.read_csv(all_speakers_fname)

	# Process rounds data
	process_rounds_data(tournaments, all_speakers)

	# Speaker data
	process_speaker_data(tournaments, all_speakers)


if __name__ == '__main__':
	main()