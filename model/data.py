import itertools
import logging
import os

import pandas as pd

from model import DATA_FOLDER
from model.loader import load_df_from_csv

logger = logging.getLogger(__name__)


def import_dataset():
    teams_df = load_df_from_csv(os.path.join(DATA_FOLDER, 'teams.csv'))
    economy_df = load_df_from_csv(os.path.join(DATA_FOLDER, 'economy.csv'))
    results_df = load_df_from_csv(os.path.join(DATA_FOLDER, 'results.csv'))
    maps_df = load_df_from_csv(os.path.join(DATA_FOLDER, 'maps.csv'))
    players_df = load_df_from_csv(os.path.join(DATA_FOLDER, 'players.csv'))
    players_form_df = load_df_from_csv(os.path.join(DATA_FOLDER, 'player_map_performance.csv'))

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

    #
    players_df = players_df[['match_id', 'team', 'player_id']]
    players_df = players_df.groupby(by=['match_id', 'team'])['player_id'].apply(list).reset_index().copy()

    # Filter for only matches with all player information
    players_df = players_df[players_df['player_id'].map(len) == 5]
    unpacked_players = pd.DataFrame(players_df['player_id'].to_list(), columns=['p1', 'p2', 'p3', 'p4', 'p5'])
    players_df[['p1', 'p2', 'p3', 'p4', 'p5']] = unpacked_players

    # Team players
    match_df = pd.merge(match_df, players_df, how='inner', left_on=['match_id', 'team_1'],
                        right_on=['match_id', 'team'])
    match_df = pd.merge(match_df, players_df, how='inner', left_on=['match_id', 'team_2'],
                        right_on=['match_id', 'team'], suffixes=['_t1', '_t2'])

    #
    players_form_df = players_form_df[["player_id", "map", "rating"]]

    for i in range(1, 6):
        for j in [1, 2]:
            player_col = f'p{i}_t{j}'

            temp = players_form_df.rename(columns={
                'map': '_map',
                'player_id': player_col
            })
            match_df = pd.merge(match_df, temp, how='inner', on=['_map', player_col])
            match_df.rename(columns={'rating': f'rating_{player_col}'}, inplace=True)

    # # Drop unneccesary columns
    # match_cols = ['team_1_id', 'team_2_id', 'rank_1', 'rank_2', 'best_of', 'map_id', 'starting_ct', 'map_wins_1', 'map_wins_2']
    # round_cols = list(itertools.chain.from_iterable(
    #     [[f'{i}_t1', f'{i}_t2', f'{i}_winner'] for i in range(1, 31)]
    # ))
    # match_df = match_df[match_cols + round_cols]
    #
    # Set types for data
    # Filter invalid 'best_of' values
    match_df = match_df.loc[match_df['best_of'] != 'o']
    # match_df.astype('int32')
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

        # All rounds played in the game with round winner and economy
        cur_rounds = df.loc[df[f'{i}_winner'] > 0, :].copy()

        # Append the current round winner to the last column
        cur_rounds['round_winner'] = cur_rounds[f"{i}_winner"].apply(lambda x: 1 if x == 1 else -1)
        cur_rounds['t1_equipment'] = cur_rounds[f'{i}_t1']
        cur_rounds['t2_equipment'] = cur_rounds[f'{i}_t2']

        # Delete the current round {i} winner in column [i_winner]
        cur_rounds[f'{i}_winner'] = 0
        if i < 30:
            cur_rounds[f'{i + 1}_t1'] = 0
            cur_rounds[f'{i + 1}_t2'] = 0

        rounds_df = rounds_df.append(cur_rounds, ignore_index=True)

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
