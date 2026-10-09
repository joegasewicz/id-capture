import json
import logging


stream_handler = logging.StreamHandler()

log = logging.getLogger(__name__)
log.addHandler(stream_handler)
log.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[ID Checker]: {message}",
    style="{",
)
stream_handler.setFormatter(formatter)


def log_json(message: str, data: dict):
    log.info(
        f"{message}:\n%s",
        json.dumps(data, indent=2, default=str),
    )
