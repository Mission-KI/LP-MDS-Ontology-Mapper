import re
from pathlib import Path

from filetype import guess


def sanitize_file_part(file_part: str) -> str:
    """Keep only alphanumeric characters and dash, underscore. Replace dot with underscore."""
    return re.sub(r"[^a-zA-Z0-9-_]", "", file_part.replace(".", "_"))


def sanitize_file_path(file_path: str) -> str:
    """Keep only alphanumeric characters and dash, underscore, slash, dot."""
    return re.sub(r"[^a-zA-Z0-9-_/.]", "", file_path)


def calculate_size(path: Path) -> int:
    """Calculate size in Bytes"""
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum((calculate_size(file) for file in path.iterdir()))
    elif path.is_symlink():
        return 0
    else:
        raise RuntimeError(f'Can not determine size of "{path}"')


def determine_file_type(path: Path) -> str:
    """Determine file type using content and extension."""
    most_likely_type = guess(path.absolute())
    if most_likely_type is not None:
        return str(most_likely_type.extension)
    _, type_by_suffix = path.name.split(".", 1)
    if type_by_suffix is not None:
        return type_by_suffix
    raise RuntimeError(f'Unable to determine type of "{path}"')
