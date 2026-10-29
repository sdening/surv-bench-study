import sys
from pathlib import Path

def use_local_survtrace():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "Model_Training" / "third_party"))