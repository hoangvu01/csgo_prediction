import os
import logging
import itertools
import pandas as pd

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from joblib import dump, load

from model import DATA_FOLDER
from model.loader import load_df_from_csv, save_processed_data, load_processed_data, save_model

logger = logging.getLogger(__name__)

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
    match_df = match_df.rename(columns={'team_id' : 'team_1_id'})
    match_df = match_df.drop(columns=['team'])
    
    # Fill in TEAM ID for team_2
    teams_df = teams_df[['team_id', 'team']]
    match_df = pd.merge(match_df, teams_df, how='inner', 
                        left_on='team_2', right_on='team')
    match_df = match_df.rename(columns={'team_id' : 'team_2_id'})
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

def pad_rounds_with_zeros(df):
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
        cur_rounds['round_winner'] = cur_rounds[f'{i}_winner']
        cur_round_winner = cur_rounds[f'{i}_winner']

        # Delete the current round winner in column [i_winner]
        cur_round_col_index = match_cols - 1 + i * 3
        cur_rounds.iloc[:, cur_round_col_index : total_cols ] = 0

        rounds_df = rounds_df.append(cur_rounds, ignore_index=True)
   
    rounds_df.astype('int32')
    rounds_df.index.name = 'id'

    logger.info("Successfully create rounds data from whole game economy")
    return rounds_df

def train(clean_slate=True):
    df = import_dataset() if clean_slate else load_processed_data(filename="prep_data")
    filter_dataset(df, min_rank=20)
    df = pad_rounds_with_zeros(df)

    save_processed_data(df, "prep_data")
    
    clf = SVC(verbose=True)

    model_inputs = df.iloc[:, :-1]
    model_outputs = df['round_winner']
    
    X_train, X_test, y_train, y_test = train_test_split(model_inputs, model_outputs, 
                                                        test_size=0.80, random_state=42)
    
    logger.info("Data preprocessing completed")
    logger.info("Beginning training...")
    model = clf.fit(X_train, y_train)

    save_model(model, "clf")

    logger.info("Training complete, testing...")
    y_prediction = model.predict(X_test)
    
    logger.info("Testing complete, analysing result...") 
    avg_score = average_precision_score(y_test, y_prediction)
    logger.info(f"Accuracy: {avg_score}")

    results_df = pd.DataFrame(y_train)
    results_df["prediction"] = pd.Series(y_prediction)
    save_processed_data(y_df, filename="results") 
