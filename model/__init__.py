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

# Fallback config
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=logFormat)


# Global constants
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "../datasets")
