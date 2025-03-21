from pathlib import Path


def check_json_file_exists(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {file_path}")
    if file_path.suffix != '.json':
        raise ValueError(f"File is not a JSON file: {file_path}")
    return True
