from pathlib import Path

def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "README.md").exists():
            return parent
    return current_file.parent.parent
