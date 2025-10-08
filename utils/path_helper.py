from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
# SRC_DIR = BASE_DIR / "src"
TEST_OUTPUTS_DIR = BASE_DIR / "test_outputs"

def resolve_path(path: str) -> Path:
    if path.startswith("@data/"):
        return DATA_DIR / path.replace("@data/", "")
    # elif path.startswith("@src/"):
    #     return SRC_DIR / path.replace("@src/", "")

    # devpath ======================================================
    elif path.startswith("@test_outputs/"):
        print('test output !')
        return TEST_OUTPUTS_DIR / path.replace("@test_outputs/", "")
    return Path(path) 
