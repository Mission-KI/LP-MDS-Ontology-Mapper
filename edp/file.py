from pathlib import Path
from filetype import guess


class File:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise RuntimeError(f'"{path}" is not a file!')
        self.path = path

    @property
    def size(self) -> int:
        """Size in Bytes"""
        return _calculate_size(self.path)

    @property
    def type(self) -> str:
        most_likely_type = guess(self.path.absolute())
        if most_likely_type is not None:
            return most_likely_type.extension
        _, type_by_suffix = self.path.name.split(".", 1)
        if type_by_suffix is not None:
            return type_by_suffix
        raise RuntimeError(f'Unable to determine type of "{self.path}"')

    def __repr__(self) -> str:
        return str(self.path.as_posix())


def _calculate_size(path: Path) -> int:
    """Calculate size in Bytes"""
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum((_calculate_size(file for file in path.iterdir())))
    elif path.is_symlink():
        return 0
    else:
        raise RuntimeError(f'Can not determine size of "{path}"')
