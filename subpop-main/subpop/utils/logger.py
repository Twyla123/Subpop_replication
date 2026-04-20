import logging
import pathlib
from datetime import datetime


def get_logger_top(name: str, debug: bool):
    REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

    initial_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # set up save path
    save_path = (
        REPO_ROOT / "results" / "logs" / f"experiment_{name.lower()}_{initial_time}.log"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(save_path),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(name)
