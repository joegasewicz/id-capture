import os


class Config:
    MIDV2020_DIR_PATH = os.environ.get("MIDV2020_DIR_PATH", None)
    DRIVING_LICENCE_PATH = os.environ.get("DRIVING_LICENCE_PATH", None)
