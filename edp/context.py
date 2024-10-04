from abc import ABC, abstractmethod
from asyncio import get_running_loop
from contextlib import asynccontextmanager
from io import TextIOWrapper
from logging import getLogger
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Tuple

from matplotlib.axes import Axes
from matplotlib.pyplot import subplots

from edp.types import FileReference


class TextWriter(ABC):
    """Abstract class for object that supports async writes with text"""

    @abstractmethod
    async def write(self, text: str) -> None: ...


class TextFileWrite(TextWriter):
    def __init__(self, io_wrapper: TextIOWrapper) -> None:
        self._wrapper = io_wrapper

    async def write(self, text: str) -> None:
        loop = get_running_loop()
        await loop.run_in_executor(None, self._wrapper.write, text)


class OutputContext(ABC):
    """Abstract class that provides functions to generate files for the service and reference them.

    Depending on the implementation, these files will be stored locally or in the cloud."""

    @abstractmethod
    @asynccontextmanager
    def get_text_file(self, name: str) -> AsyncIterator[Tuple[TextWriter, FileReference]]: ...

    @abstractmethod
    @asynccontextmanager
    def get_plot(self, name: str) -> AsyncIterator[Tuple[Axes, FileReference]]: ...


class OutputLocalFilesContext(OutputContext):
    """This supplies functions to generate output files and graphs."""

    def __init__(self, path: Path, encoding: str = "utf-8") -> None:
        self._logger = getLogger(__name__)
        if path.exists() and not path.is_dir():
            self._logger.info('Output path "%s" must be a directory!', path)
        if not path.exists():
            self._logger.info('Creating output path "%s"', path)
            path.mkdir(parents=True)
        self.path = path
        self.encoding = encoding

    @asynccontextmanager
    async def get_text_file(self, name: str):
        path = self.path / name
        with open(path, "wt", encoding=self.encoding) as io_wrapper:
            yield TextFileWrite(io_wrapper), PurePosixPath(path)
        self._logger.debug('Generated text file "%s"', path)

    @asynccontextmanager
    async def get_plot(self, name: str):
        save_path = self.path / name
        if not save_path.suffix:
            save_path = save_path.with_suffix(".png")
        if save_path.exists():
            self._logger.warning('The path "%s" already exists, will overwrite!', save_path)
            save_path.unlink()
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = subplots()
        yield axes, PurePosixPath(save_path.relative_to(self.path))
        figure.savefig(save_path)
        self._logger.debug('Generated plot "%s"', save_path)
