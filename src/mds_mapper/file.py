import re
from pathlib import Path


def sanitize_file_part(file_part: str) -> str:
    """Keep only alphanumeric characters and dash, underscore. Replace dot with underscore."""
    return re.sub(r"[^a-zA-Z0-9-_]", "", file_part.replace(".", "_"))


def sanitize_path(path: str) -> str:
    """Keep only alphanumeric characters and dash, underscore, dot and slash."""
    return re.sub(r"[^a-zA-Z0-9-_/.]", "", path)


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


def build_real_sub_path(base_path: Path, sub_path: Path | str) -> Path:
    """Combine base_path and sub_path, enforcing that the resulting path is a sub path of base_path."""
    full_path = base_path / sub_path
    if not is_real_subpath(full_path, base_path):
        raise ValueError(f"{sub_path} escapes the base path {base_path}")
    return full_path


def is_real_subpath(path: Path, base_path: Path):
    """Check if path is a real subpath of base_path."""
    path_resolved = path.resolve()
    base_path_resolved = base_path.resolve()
    return path_resolved != base_path_resolved and path_resolved.is_relative_to(base_path_resolved)
