from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "generate_data.py"


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
