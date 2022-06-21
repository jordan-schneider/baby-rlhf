import json
from pathlib import Path
from typing import List, Tuple


def load_hh_data(path: Path) -> List[Tuple[str, str]]:
    """
    Loads the HH data from the given path.
    """
    with open(path, "r") as f:
        out = []
        for line in f.readlines():
            data = json.loads(line)
            out.append((data["chosen"], data["rejected"]))
    return out
