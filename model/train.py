import logging

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from model.data import import_dataset, filter_dataset, split_economy_into_rounds, create_custom_features
from model.loader import save_processed_data, save_model

logger = logging.getLogger(__name__)


def train(clf, clf_name, inputs, outputs, test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_size)

    logger.info("Data preprocessing completed")
    logger.info("Beginning training...")
    logger.info(f"Model - {clf_name} TestSize - {test_size}")
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
        "training size": len(x_train),
        "testing size": len(x_test),
        "training set accuracy": y_train_score,
        "test set accuracy": y_test_score
    }


def train_multi_model():
    df = import_dataset()
    df = filter_dataset(df, min_rank=20)
    df = split_economy_into_rounds(df)
    df = create_custom_features(df)

    # Select only required columns
    df = df[
        ['team_1_id', 'team_2_id', 'rank_1', 'rank_2', 'best_of', 'map_id', 'ct_team',
         'map_wins_1', 'map_wins_2', 't1_score', 't2_score', 't1_equipment', 't2_equipment',
         't1_streak', 't2_streak', 'rating_p1_t1', 'rating_p1_t2', 'rating_p2_t1', 'rating_p2_t2',
         'rating_p3_t1', 'rating_p3_t2', 'rating_p4_t1', 'rating_p4_t2', 'rating_p5_t1', 'rating_p5_t2',
         'round_winner']
    ]

    save_processed_data(df, "prep_data")

    logger.info(df.columns)

    classifier_names = ["Nearest Neighbors",
                        "Decision Tree", "Random Forest", "Neural Net",
                        "AdaBoost", "Naive Bayes", "QDA",
                        # "Linear SVM", "RBF SVM"
                        ]
    classifiers = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
    ]

    inputs = df.iloc[:, :-1]
    outputs = df['round_winner']

    return [
        train(clf, classifier_names[i], inputs, outputs, test_size=p)
        for i, clf in enumerate(classifiers) for p in [0.3, 0.5, 0.7]]
