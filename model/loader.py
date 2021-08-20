import os
import glob
import logging
import pandas as pd

from datetime import datetime
from model import PROCESSED_FOLDER

logger = logging.getLogger(__file__)

def load_df_from_csv(abspath):
    if not os.path.exists(abspath):
        logger.error(f"File does not exist, 'abspath': {abspath}")
        raise Exception("Invalid file exception")
    if not os.path.isfile(abspath):
        logger.error(f"'abspath' must be a file, provided: {abspath}")
        raise Exception("Invalid file exception")

    logger.info(f"Importing file {abspath} into pandas dataframe")
    return pd.read_csv(abspath, low_memory=False)

def load_processed_data():
    logger.info("Attempting to load most recent CSV file into dataframe")

    paths = os.path.join(PROCESSED_FOLDER, '*.csv')
    files = glob.glob(paths)
    logger.info(f"Found the following files {files}")

    latest_file = max(files)
    return load_df_from_csv(latest_file)


def save_processed_data(df, timestamp=True):
    filename = "prep_data"

    if timestamp:
        filename += datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")

    if not os.path.exists(PROCESSED_FOLDER):
        os.mkdir(PROCESSED_FOLDER)

    path = os.path.join(PROCESSED_FOLDER, filename)
    logger.info(f'Saving processed data to {path}')
    df.to_csv(path, )
    logger.info(f'Data has been saved to {path}')
