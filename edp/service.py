from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import AsyncIterator, Dict, List, Optional, Set

from edp.compression import CompressionAlgorithm
from edp.file import File
from edp.importers import Importer, csv_importer
from edp.types import Compression, ComputedAssetData, Dataset, DataSetType


class Service:
    def __init__(self):
        self._logger = getLogger(__name__)
        self._logger.info("Initializing")

        self._importers = _create_importers()
        self._compressions = _create_compressions()
        self._logger.info("The following data types are supported: [%s]", ", ".join(self._importers))
        implemented_compressions = [key for key, value in self._compressions.items() if value is not None]
        self._logger.info("The following compressions are supported: [%s]", ", ".join(implemented_compressions))

    async def analyse_asset(self, path: Path) -> ComputedAssetData:
        if not path.exists():
            raise FileNotFoundError(f'File "{path}" can not be found!')
        if not path.is_file():
            raise RuntimeError("Please pass the path to a single file!")
        file = File(path)
        compressions: List[str] = []
        extracted_size = 0
        datasets: List[Dataset] = []
        data_structures: Set[DataSetType] = set()

        async for child_files in self._walk_all_files(file.path, compressions):
            file_type = child_files.type
            extracted_size += child_files.size
            if not file_type in self._importers:
                raise NotImplementedError(f'Import for "{file_type}" not yet implemented')
            structure = await self._importers[file_type](child_files)
            data_structures.add(structure.data_set_type)
            datasets.append(await structure.analyze())

        compression: Optional[Compression]
        if len(compressions) == 0:
            compression = None
        else:
            compression = Compression(algorithms=compressions, extractedSize=extracted_size)

        return ComputedAssetData(
            volume=file.size,
            compression=compression,
            dataTypes=data_structures,
            datasets=datasets,
        )

    async def _walk_all_files(self, path: Path, compressions: List[str]) -> AsyncIterator[File]:
        """Will yield all files, recursively through directories and archives."""

        if path.is_file():
            file = File(path)
            if file.type not in self._compressions:
                yield file
            else:
                compressions.append(file.type)
                async with self._extract(file) as extracted_path:
                    async for child_file in self._walk_all_files(extracted_path, compressions):
                        yield child_file
        elif path.is_dir():
            for file_path in path.iterdir():
                async for child_file in self._walk_all_files(file_path, compressions):
                    yield child_file
        else:
            self._logger.warning('Can not extract or analyse "%s"', path)

    @asynccontextmanager
    async def _extract(self, file: File) -> AsyncIterator[Path]:
        archive_type = file.type
        if not archive_type in self._compressions:
            raise RuntimeError(f'"{archive_type}" is not a know archive type')
        compression = self._compressions[archive_type]
        if compression is None:
            raise NotImplementedError(f'Extractin "{archive_type}" is not implemented')
        with TemporaryDirectory() as directory_str:
            directory = Path(directory_str)
            await compression.extract(file.path, directory)
            yield directory


def _create_importers() -> Dict[str, Importer]:
    return {"csv": csv_importer}


def _create_compressions() -> Dict[str, Optional[CompressionAlgorithm]]:
    return {
        "br": None,
        "rpm": None,
        "dcm": None,
        "epub": None,
        "zip": None,
        "tar": None,
        "rar": None,
        "gz": None,
        "bz2": None,
        "7z": None,
        "xz": None,
        "pdf": None,
        "exe": None,
        "swf": None,
        "rtf": None,
        "eot": None,
        "ps": None,
        "sqlite": None,
        "nes": None,
        "crx": None,
        "cab": None,
        "deb": None,
        "ar": None,
        "Z": None,
        "lzo": None,
        "lz": None,
        "lz4": None,
        "zstd": None,
    }
