import os
import logging
import itertools
import pandas as pd

from model import DATA_FOLDER
from model.loader import load_df_from_csv

logger = logging.getLogger(__name__)

def create_teams_csv():
    results_path = os.path.abspath(os.path.join(DATA_FOLDER, "results.csv"))
    results_df = pd.read_csv(results_path)

    # Removes all but specified columns
    results_df = results_df[['date', 'team_1', 'team_2', 'rank_1', 'rank_2']]

    # Combines the columns to get TEAM and RANK
    teams_df = pd.concat([
        results_df[['date', 'team_1', 'rank_1']], 
        results_df[['date', 'team_2', 'rank_2']]    
    ])

    # Cleanup redundant and/or empty columns
    teams_df = teams_df.drop(columns=['team_2', 'rank_2'])
    teams_df = teams_df.rename(columns={'team_1': 'team', 'rank_1': 'rank'})
    teams_df = teams_df.dropna().drop_duplicates()

    # Convert ranking from float32 to int32
    teams_df = teams_df.astype({'rank': 'int32'})

    # Rank teams based on their most recent rank recorded
    teams_df = teams_df.sort_values('date', ascending=False)
    teams_df = teams_df.groupby('team').first()
    teams_df = teams_df.drop(columns=['date'])
    teams_df = teams_df.sort_values('rank')

    logger.info(f'Data aggregation completed,{len(teams_df)} teams found.')
    logger.debug(teams_df.head())

    # Name ID column title
    teams_df = teams_df.reset_index()
    teams_df.index = teams_df.index + 1
    teams_df.index.name = 'team_id'

    # Write results to csv file
    output_path = os.path.abspath(os.path.join(DATA_FOLDER, "teams.csv"))
    logger.info(f'Writing results to {output_path}')
    teams_df.to_csv(output_path)
    logger.info(f'Write operation to {output_path} completed')
   
def create_maps_csv():
    results_path = os.path.abspath(os.path.join(DATA_FOLDER, "results.csv"))
    results_df = pd.read_csv(results_path)

    # Create unique ID for each map 
    maps = set(results_df['_map'].tolist())
    maps_id = [ [k, v] for k,v in zip(maps, range(len(maps))) ]
    maps_df = pd.DataFrame(maps_id, columns=['map', 'map_id'])
    maps_df.set_index('map_id', inplace=True)

    # Write results to csv file
    output_path = os.path.abspath(os.path.join(DATA_FOLDER, "maps.csv"))
    logger.info(f'Writing results to {output_path}')
    maps_df.to_csv(output_path)
    logger.info(f'Write operation to {output_path} completed')


def import_dataset():
    teams_df = load_df_from_csv(os.path.join(DATA_FOLDER, 'teams.csv'))
    economy_df = load_df_from_csv(os.path.join(DATA_FOLDER, 'economy.csv'))
    results_df = load_df_from_csv(os.path.join(DATA_FOLDER, 'results.csv'))
    maps_df = load_df_from_csv(os.path.join(DATA_FOLDER, 'maps.csv'))

    # Inner join the 2 dataframes
    match_df = pd.merge(
        economy_df,
        results_df,
        how='inner',
        on=['match_id', 'event_id', 'team_1', 'team_2', '_map']
    )
    match_df.fillna(0, inplace=True)

    # Fill in TEAM ID for team_1
    teams_df = teams_df[['team_id', 'team']]
    match_df = pd.merge(match_df, teams_df, how='inner',
                        left_on='team_1', right_on='team')
    match_df = match_df.rename(columns={'team_id': 'team_1_id'})
    match_df = match_df.drop(columns=['team'])

    # Fill in TEAM ID for team_2
    teams_df = teams_df[['team_id', 'team']]
    match_df = pd.merge(match_df, teams_df, how='inner',
                        left_on='team_2', right_on='team')
    match_df = match_df.rename(columns={'team_id': 'team_2_id'})
    match_df = match_df.drop(columns=['team'])

    # Replace MAP with MAP_ID
    match_df = pd.merge(match_df, maps_df, how='inner',
                        left_on='_map', right_on='map')

    # Drop unneccesary columns
    match_cols = ['team_1_id', 'team_2_id', 'rank_1', 'rank_2', 'best_of', 'map_id', 'starting_ct']
    round_cols = list(itertools.chain.from_iterable(
        [[f'{i}_t1', f'{i}_t2', f'{i}_winner'] for i in range(1, 31)]
    ))
    match_df = match_df[match_cols + round_cols]

    # Set types for data
    # Filter invalid 'best_of' values
    match_df = match_df.loc[match_df['best_of'] != 'o']
    match_df.astype('int32')
    match_df.index.title = 'id'

    return match_df


def filter_dataset(df, min_rank=30, best_of=None):
    df = df[(df.rank_1 < min_rank) & (df.rank_2 < min_rank)]

    if not (best_of is None):
        df = df[df.best_of == best_of]

    return df


def split_economy_into_rounds(df):
    logger.info("Augmenting data by serialising round economy...")

    columns = list(df) + ['round_winner']
    rounds_df = pd.DataFrame(columns=list(df))
    total_cols = len(df.columns)
    round_cols = 90
    match_cols = total_cols - round_cols

    for i in range(30, 0, -1):
        logger.info(f"Creating data for the {i} round.")

        # All rounds plated in the game with round winner and economy
        cur_rounds = df.loc[df[f'{i}_winner'] > 0, :].copy()

        # Append the current round winner to the last column
        cur_rounds['round_winner'] = cur_rounds[f"{i}_winner"].apply(lambda x: 1 if x == 1 else -1)
        cur_rounds['t1_equipment'] = cur_rounds[f'{i}_t1']
        cur_rounds['t2_equipment'] = cur_rounds[f'{i}_t2']

        # Delete the current round winner in column [i_winner]
        cur_round_col_index = match_cols - 1 + i * 3
        cur_rounds.iloc[:, cur_round_col_index: total_cols] = 0

        rounds_df = rounds_df.append(cur_rounds, ignore_index=True)

    rounds_df.astype('int32')
    rounds_df.index.name = 'id'

    logger.info("Successfully create rounds data from whole game economy")
    return rounds_df

def create_custom_features(df):
    winner_cols = [f"{i}_winner" for i in range(1, 31)]
    winners = df[winner_cols]

    df["t1_score"] = winners.apply(current_score, axis=1, team_number=1)
    df["t2_score"] = winners.apply(current_score, axis=1, team_number=2)

    df["t1_streak"] = winners.apply(current_streak, axis=1, team_number=1)
    df["t2_streak"] = winners.apply(current_streak, axis=1, team_number=2)

    df['ct_team'] = df.apply(label_ct, axis=1)

    return df

def label_ct(row):
    if row["t1_score"] + row["t2_score"] > 15:
        return 3 - row["starting_ct"]
    return row["starting_ct"]

def current_score(row, team_number):
    return len([i for i in row.tolist() if i == team_number])

def current_streak(row, team_number):
    row_list = row.tolist()
    row_list.reverse()
    round_winners = itertools.dropwhile(lambda x: x == 0, row_list)

    streak = itertools.takewhile(lambda x: x == team_number, round_winners)
    return len(list(streak))