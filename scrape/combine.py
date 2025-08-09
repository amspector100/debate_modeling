import time
import numpy as np
import pandas as pd
import glob
import copy
import os
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

#### Minor helper functions
# process side names
SIDE_NAMES = {
	"CG":"CG",
	"CO":"CO",
	"OG":"OG",
	"OO":"OO",
	"Closing Government":"CG",
	"Closing Opposition":"CO",
	"Opening Government":"OG",
	"Opening Opposition":"OO",
}
RESULT2RANK = {
	"1st":3,
	"2nd":2,
	"3rd":1,
	"4th":0,
	"advancing":3,
	"eliminated":0,
}
def _map_column_verbose(df, col, map_dict, newcol=None):
	""" sets df[newcol] = df[col].map(map_dict) but warns if produces NaNs """
	flags = df[col].isin(map_dict.keys())
	if not np.all(flags):
		raise ValueError(f"Unrecognized side names = {df.loc[flags, col].unique()}, add to {map_dict}.")
	if newcol is None:
		newcol = col
	df[newcol] = df[col].map(map_dict)
	return df

def _combine_multicols(df):
	newcols = df.columns.get_level_values(1)
	newcols = newcols + "_" + df.columns.get_level_values(0)
	df.columns = newcols
	#df = df[sorted(newcols)]
	return df

#### Major processing functions
def create_team2teamid(tourn_id):
	"""
	Returns
	-------
	teams2ids : dict
		Maps team to id. The ID is unique within a tournament
		but not between tournaments.
	"""
	# Read
	tabdf = pd.read_csv(f"../data/{tourn_id}/raw/tab.csv", sep='\t')
	tabdf.set_index("team", inplace=True)
	# convert to dict
	tabdf['team_id'] = np.arange(len(tabdf)).astype(int)
	return tabdf['team_id'].to_dict()

def clean_speaker_df(sdf, all_speakers=None):
	# Correctly process names: replace two periods with one
	sdf = sdf.loc[sdf['name'].notnull()].copy()
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
		print(f"Reading speakers from {tourndir}.")
		sdf = pd.read_csv(tourndir + "speakers.csv", sep='\t')
		sdf = clean_speaker_df(sdf)

		dup_flags = sdf.duplicated("name", keep=False)
		dup_flags = dup_flags & (sdf['name'] != "swing_speaker")
		if np.any(dup_flags):
			print(f"\nAt {code}, some duplicated speaker names:\n{sdf.loc[dup_flags]}")
			print("By default, treating all of these as the same person.")
			sdf = sdf.loc[~sdf.duplicated("name", keep='first')].copy()

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

def process_single_round(
	rdf, 
	team2speakerid,
	team2teamid,
	swing_id,
	all_speaks,
	tourn_id,
	round_number
):
	# team id
	rdf['team_id'] = rdf['team'].map(team2teamid)
	rdf['team_id'] = rdf['team_id'].fillna(-1)

	# Missing teams get mapped to the swing speaker
	rteams = set(rdf['team'].unique().tolist())
	steams = set(team2speakerid.index.tolist())
	diff = rteams - steams
	if len(diff) > 0:
		for d in diff:
			team2speakerid.loc[d] = [swing_id]

	# Find speaker ids
	sids = team2speakerid.loc[rdf['team']]
	rdf['speaker_id0'] = sids.apply(lambda x: x[0]).values
	rdf['speaker_id1'] = sids.apply(lambda x: x[-1]).values

	# for inrounds, ensure there is at exactly one 1st, 2nd, 3rd, 4th per round
	if '1st' in rdf['result'].unique():
		counts = rdf.groupby(["adjudicators", "result"])['team'].count().reset_index()
		bad_rds = counts.loc[counts['team'] != 1, 'adjudicators'].unique()
		rdf = rdf.loc[~rdf['adjudicators'].isin(bad_rds)]
		# Ensure there are 4 teams per round
		counts = rdf.groupby(['adjudicators'])['team'].count().reset_index()
		bad_rds = counts.loc[counts['team'] != 4, 'adjudicators'].unique()
		rdf = rdf.loc[~rdf['adjudicators'].isin(bad_rds)]

	# clean side names and results
	rdf = _map_column_verbose(rdf, col='side', map_dict=SIDE_NAMES)
	rdf = _map_column_verbose(rdf, col='result', newcol='rank', map_dict=RESULT2RANK)

	# Merge with speaks data for inrounds
	speaksub = all_speaks.loc[
		(all_speaks['tourn_id'] == tourn_id) &
		(all_speaks['round'].astype(int) == int(round_number))
	]
	if len(speaksub) == 0:
		rdf['speaks0'] = np.nan
		rdf['speaks1'] = np.nan
	else:
		for sid in ['0', '1']:
			# change names to make merging easier
			on = [f'speaker_id{sid}', 'team_id']
			speaksub = speaksub.rename(columns={"id":f"speaker_id{sid}", "speaks":f"speaks{sid}"})
			intermed = pd.merge(
				rdf, speaksub[on + [f'speaks{sid}']], how='left'
			)
			if sid == '0':
				intermed = intermed.groupby(on)[f'speaks{sid}'].first()
			else:
				intermed = intermed.groupby(on)[f'speaks{sid}'].last()
			rdf = pd.merge(rdf, intermed, on=on, how='left', validate='one_to_one')
			# reset speaksub
			speaksub = speaksub.rename(columns={f"speaker_id{sid}":"id", f"speaks{sid}":"speaks"})
			

	# pivot to longer format
	rdf = rdf[['rank', 'side', 'team_id', 'adjudicators', 'speaker_id0', 'speaker_id1', 'speaks0', 'speaks1']].pivot(
		index='adjudicators', columns='side'
	)
	rdf = _combine_multicols(rdf)
	# simplify index
	rdf.index = np.arange(len(rdf))
	return rdf

def process_rounds_data(tournaments, all_speakers, all_speaks):
	"""
	Assumes that all_speakers, all_speaks have already been created
	"""
	swing_id = all_speakers.loc[
		all_speakers['name'].str.contains('speaker'), 'id'
	].item()

	# Initialize data output
	all_data = []
	for ii in range(len(tournaments)):
		code = tournaments.iloc[ii]['Code']
		date = tournaments.iloc[ii]['Date']
		print(f"At code={code}.")
		# Load speaker data
		tourndir = f"../data/{code}/raw/"
		sdf = pd.read_csv(tourndir + "speakers.csv", sep='\t')
		sdf = clean_speaker_df(sdf, all_speakers=all_speakers)
		# Map teams to speakers
		team2speakerid = sdf.groupby("team")['id'].apply(
			lambda x: list(np.unique(x))
		)
		team2teamid = create_team2teamid(code)
		# Load round data
		rfiles = sorted(glob.glob(tourndir + "round*.csv"))
		for rfile in rfiles:
			# check this is really round results
			round_number = rfile.split("round")[-1].split(".csv")[0]
			if is_float(round_number):
				rdf = pd.read_csv(rfile, sep='\t')
				try:
					rdf = process_single_round(
						rdf=rdf,
						team2speakerid=team2speakerid,
						team2teamid=team2teamid,
						swing_id=swing_id,
						all_speaks=all_speaks,
						tourn_id=code,
						round_number=round_number,
					)
				except Exception as e:
					print(f"Error processing round {round_number} at {code}: {e}")
					print(f"rdf head: {rdf.head()}")
					raise e					
				rdf['tourn_id']= code
				rdf['date'] = date
				rdf['round_number'] = round_number
				all_data.append(rdf)

	round_df = pd.concat(all_data)
	round_df.to_csv("../data/combined/round_data.csv", index=False)
	return round_df

def process_speaker_data(tournaments, all_speakers):
	# swing_id = all_speakers.loc[
	# 	all_speakers['name'].str.contains('speaker'), 'id'
	# ].item()
	# Initialize data output
	all_winners = []
	all_participants = []
	sdfs = []
	for cid, code in enumerate(tournaments['Code']):
		# team --> team id
		team2teamid = create_team2teamid(code)
		# Load speaker data
		tourndir = f"../data/{code}/raw/"
		sdf = pd.read_csv(tourndir + "speakers.csv", sep='\t')
		sdf = clean_speaker_df(sdf, all_speakers=all_speakers).copy()
		sdf['tourn_id'] = code
		sdf['team_id'] = sdf['team'].map(team2teamid)
		sdf['team_id'] = sdf['team_id'].fillna(-1)
		sdfs.append(sdf)

	# Combine
	sdf = pd.concat(sdfs, axis='index')
	sdf[sdf == '—'] = np.nan
	# infer which columns are speaks for rounds
	ids = ['id', 'tourn_id', 'team_id']
	rel_cols = copy.copy(ids)
	for c in sdf.columns:
		cl = c.lower()
		if cl[0] == 'r' and is_float(cl.split('r')[-1]):
			rel_cols.append(c)
	sdf = sdf[rel_cols].melt(id_vars=ids, var_name='round', value_name='speaks')
	sdf = sdf.loc[sdf['speaks'].notnull()].copy()
	sdf['round'] = sdf['round'].apply(lambda x: x[1:]).astype(int)
	sdf.to_csv("../data/combined/all_speaks.csv", index=False)
	return sdf

def main():
	t0 = time.time()

	# Read list of tournaments
	tournaments = pd.read_csv("../data/tournaments.csv")

	# Create global speaker list
	create_global_speaker_list(tournaments)

	# Read list of speakers
	all_speakers = pd.read_csv(all_speakers_fname)

	# Speaker data
	all_speaks = process_speaker_data(tournaments, all_speakers)

	# Process rounds data
	process_rounds_data(tournaments, all_speakers, all_speaks)

if __name__ == '__main__':
	os.makedirs("../data/combined", exist_ok=True)
	main()