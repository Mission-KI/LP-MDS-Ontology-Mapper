from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from filetype import guess
from matplotlib.axes import Axes
from matplotlib.pyplot import subplots


class OutputContext:
    """This supplies functions to generate output files and graphs."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get_sub_dir(self, path: Path) -> Path:
        resulting = self.path / path
        resulting.mkdir(exist_ok=True)
        return resulting

    @asynccontextmanager
    async def create_plot(self, path: Path) -> AsyncIterator[Axes]:
        save_path = self.path / path
        if not save_path.suffix:
            save_path /= ".png"
        if save_path.exists():
            raise RuntimeError(f'Can not create plot, the file "{save_path}" already exists!')
        figure, axes = subplots()
        yield axes
        figure.savefig(save_path)


class File:
    def __init__(self, base_path: Path, path: Path) -> None:
        if not path.is_file():
            raise RuntimeError(f'"{path}" is not a file!')
        if not base_path.is_dir():
            raise RuntimeError(f'base_path "{base_path}" must be a directory!')
        if not path.is_relative_to(base_path):
            raise RuntimeError(f'"{path}" must be inside "{base_path}"!')
        self._base_path = base_path
        self.path = path

    @property
    def size(self) -> int:
        """Size in Bytes"""
        return self.path.stat().st_size

    @property
    def type(self) -> str:
        most_likely_type = guess(self.path.absolute())
        if most_likely_type is not None:
            return str(most_likely_type.extension)
        _, type_by_suffix = self.path.name.split(".", 1)
        if type_by_suffix is not None:
            return type_by_suffix
        raise RuntimeError(f'Unable to determine type of "{self.path}"')

    @property
    def relative(self) -> Path:
        return self.path.relative_to(self._base_path)

    def __repr__(self) -> str:
        return str(self.relative.as_posix())


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
