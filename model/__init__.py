import os
import sys
import logging

# Logger global configurations
root = logging.getLogger(__name__)
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logFormat = "%(asctime)s - %(name)s - [%(levelname)s]: %(message)s"
formatter = logging.Formatter(logFormat)
handler.setFormatter(formatter)
root.addHandler(handler)


# Global constants
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets"))
PROCESSED_FOLDER = os.path.join(DATA_FOLDER, "processed")
