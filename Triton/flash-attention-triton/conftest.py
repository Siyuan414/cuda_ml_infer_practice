"""Makes the package importable when running `pytest` without `pip install -e .`."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
