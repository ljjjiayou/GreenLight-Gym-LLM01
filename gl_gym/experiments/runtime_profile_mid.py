import json
import sys
from pathlib import Path

cur_dir = Path(__file__).resolve().parent
if str(cur_dir) not in sys.path:
    sys.path.insert(0, str(cur_dir))

from runtime_profile_quick import run_case


if __name__ == "__main__":
    out = [
        run_case("mid_i30_t160", 30, 160, max_steps=60, seed=42),
        run_case("mid_i45_t128", 45, 128, max_steps=60, seed=42),
    ]
    print(json.dumps({"profile": out}, ensure_ascii=False, indent=2))
