from pathlib import Path
import time


def pytest_configure(config):
    root = Path(__file__).resolve().parents[1]
    (root/'tmp').mkdir(exist_ok=True)
    if config.option.basetemp is None:
        config.option.basetemp = str(root/'tmp'/f'pytest-{time.time_ns()}')
