import logging
import os
import pandas as pd

from model import DATA_FOLDER
from model.train import import_dataset

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
