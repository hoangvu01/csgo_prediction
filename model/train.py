import itertools
import logging
import os
import pandas as pd
from joblib import dump, load
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

        cur_round_winner = cur_rounds[f'{i}_winner']

        # Delete the current round winner in column [i_winner]
        cur_round_col_index = match_cols - 1 + i * 3
        cur_rounds.iloc[:, cur_round_col_index: total_cols] = 0

        rounds_df = rounds_df.append(cur_rounds, ignore_index=True)

    rounds_df.astype('int32')
    rounds_df.index.name = 'id'

    winner_cols = [f"{i}_winner" for i in range(1, 31)]
    winners = rounds_df[winner_cols]

    rounds_df["t1_score"] = winners.apply(lambda row: len([i for i in row.tolist() if i == 1]), axis=1)
    rounds_df["t2_score"] = winners.apply(lambda row: len([i for i in row.tolist() if i == 2]), axis=1)

    logger.info("Successfully create rounds data from whole game economy")
    return rounds_df


def train(clf, clf_name, inputs, outputs, clean_slate=True, test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_size)

    logger.info("Data preprocessing completed")
    logger.info("Beginning training...")
    model = clf.fit(x_train, y_train)

    save_model(model, clf_name)

    y_train_prediction = model.predict(x_train)
    y_train_score = average_precision_score(y_train, y_train_prediction)
    logger.info(f"Testing with training set finished with {y_train_score} accuracy.")

    y_test_prediction = model.predict(x_test)
    y_test_score = average_precision_score(y_test, y_test_prediction)
    logger.info(f"Testing with test set finished with {y_test_score} accuracy.")

    results_df = pd.DataFrame(x_train)
    results_df["expected"] = pd.Series(y_train)
    results_df["prediction"] = pd.Series(y_train_prediction)
    save_processed_data(results_df, filename=clf_name + "_results_train")

    results_df = pd.DataFrame(x_test)
    results_df["expected"] = pd.Series(y_test)
    results_df["prediction"] = pd.Series(y_test_prediction)
    save_processed_data(results_df, filename=clf_name + "_results_test")

    return {
        "classifier": clf_name,
        "test size": test_size,
        "training set accuracy": y_train_score,
        "test set accuracy": y_test_score
    }


def train_multi_model():
    df = import_dataset()
    df = filter_dataset(df, min_rank=20)
    df = split_economy_into_rounds(df)

    # Select only required columns
    df = df[
        ['team_1_id', 'team_2_id', 'rank_1', 'rank_2', 'best_of', 'map_id', 'starting_ct',
         't1_score', 't2_score', 't1_equipment', 't2_equipment', 'round_winner']
    ]

    save_processed_data(df, "prep_data")

    classifier_names = [
        # "Nearest Neighbors", "Linear SVM", "RBF SVM",
        "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]
    classifiers = [
        # KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]
    classifier_res = []

    inputs = df.iloc[:, :-1]
    outputs = df['round_winner']

    return [
        train(clf, classifier_names[i], inputs, outputs, test_size=p)
        for i, clf in enumerate(classifiers) for p in [0.3, 0.5, 0.7]]
