import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

DATA_FOLDER = os.path.abspath("../datasets")


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
    maps_id = [[k, v] for k, v in zip(maps, range(len(maps)))]
    maps_df = pd.DataFrame(maps_id, columns=['map', 'map_id'])
    maps_df.set_index('map_id', inplace=True)

    # Write results to csv file
    output_path = os.path.abspath(os.path.join(DATA_FOLDER, "maps.csv"))
    logger.info(f'Writing results to {output_path}')
    maps_df.to_csv(output_path)
    logger.info(f'Write operation to {output_path} completed')


def create_player_performance_csv():
    players_path = os.path.abspath(os.path.join(DATA_FOLDER, "players.csv"))
    players_df = pd.read_csv(players_path)

    static_cols = ['player_name', 'player_id']

    map_stats_cols = ['kills', 'assists', 'deaths', 'hs', 'flash_assists',
                      'kast', 'kddiff', 'adr', 'fkdiff', 'rating']

    df = pd.DataFrame(columns=static_cols + ['map'] + map_stats_cols)

    for map_number in [1, 2, 3]:
        # Filter data for available map choice
        map_df = players_df.loc[pd.notnull(players_df[f'map_{map_number}'])]

        # Select player stats for a single map in the serie
        indexed_map_cols = [f"m{map_number}_{col}" for col in map_stats_cols]
        map_df = map_df[static_cols + [f"map_{map_number}"] + indexed_map_cols]

        # Rename columns according to returned dataframe format
        map_df = map_df.rename(columns={f"map_{map_number}": "map"})
        map_df = map_df.rename(columns={f"m{map_number}_{col}": col for col in map_stats_cols})

        df = pd.concat([df, map_df])

    df = df.dropna()
    df = df.astype({col: 'float64' for col in map_stats_cols})
    df = df.groupby(by=['player_name', 'player_id', 'map'], as_index=False).mean()

    # Write results to csv file
    output_path = os.path.abspath(os.path.join(DATA_FOLDER, "player_map_performance.csv"))
    logger.info(f'Writing results to {output_path}')
    df.to_csv(output_path)
    logger.info(f'Write operation to {output_path} completed')

