from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import AsyncIterator, Dict, Optional, Set

from pydantic import BaseModel

from edp.compression import CompressionAlgorithm
from edp.context import OutputContext
from edp.file import File, calculate_size
from edp.importers import Importer, csv_importer, pickle_importer
from edp.types import (
    Asset,
    Compression,
    ComputedAssetData,
    DataSetType,
    StructuredDataSet,
    UserProvidedAssetData,
)


class Service:
    def __init__(self):
        self._logger = getLogger(__name__)
        self._logger.info("Initializing")

        self._importers = _create_importers()
        self._compressions = _create_compressions()
        self._logger.info("The following data types are supported: [%s]", ", ".join(self._importers))
        implemented_compressions = [key for key, value in self._compressions.items() if value is not None]
        self._logger.info("The following compressions are supported: [%s]", ", ".join(implemented_compressions))

    async def analyse_asset(self, path: Path, user_data: UserProvidedAssetData, output_context: OutputContext) -> None:
        computed_data = await self._compute_asset(path, output_context)
        asset = Asset(**_as_dict(computed_data), **_as_dict(user_data))
        json_name = user_data.id + ("_" + user_data.version if user_data.version else "")
        json_name = json_name.replace(".", "_")
        json_name += ".json"
        async with output_context.get_text_file(json_name) as (output, _):
            await output.write(asset.model_dump_json())

    async def _compute_asset(self, path: Path, output_context: OutputContext) -> ComputedAssetData:
        if not path.exists():
            raise FileNotFoundError(f'File "{path}" can not be found!')
        compressions: Set[str] = set()
        extracted_size = 0
        datasets: Dict[str, StructuredDataSet] = {}
        data_structures: Set[DataSetType] = set()
        base_path = path if path.is_dir() else path.parent

        async for child_files in self._walk_all_files(base_path, path, compressions):
            file_type = child_files.type
            extracted_size += child_files.size
            if not file_type in self._importers:
                raise NotImplementedError(f'Import for "{file_type}" not yet implemented')
            structure = await self._importers[file_type](child_files)
            data_structures.add(structure.data_set_type)
            dataset_result = await structure.analyze(output_context)
            if not isinstance(dataset_result, StructuredDataSet):
                raise NotImplementedError(f'Did not expect dataset type "{type(dataset_result)}"')
            datasets[str(child_files)] = dataset_result

        compression: Optional[Compression]
        if len(compressions) == 0:
            compression = None
        else:
            compression = Compression(algorithms=compressions, extractedSize=extracted_size)

        return ComputedAssetData(
            volume=calculate_size(path),
            compression=compression,
            dataTypes=data_structures,
            structuredDatasets=datasets,
        )

    async def _walk_all_files(self, base_path: Path, path: Path, compressions: Set[str]) -> AsyncIterator[File]:
        """Will yield all files, recursively through directories and archives."""

        if path.is_file():
            file = File(base_path, path)
            if file.type not in self._compressions:
                yield file
            else:
                compressions.add(file.type)
                async with self._extract(file) as extracted_path:
                    async for child_file in self._walk_all_files(base_path, extracted_path, compressions):
                        yield child_file
        elif path.is_dir():
            for file_path in path.iterdir():
                async for child_file in self._walk_all_files(base_path, file_path, compressions):
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
        archive_name = file.path.name
        directory = file.path.parent / archive_name.replace(".", "_")
        while directory.exists():
            directory /= "extracted"

        directory.mkdir()
        await compression.extract(file.path, directory)
        try:
            yield directory
        finally:
            rmtree(directory.absolute())


def _create_importers() -> Dict[str, Importer]:
    return {"csv": csv_importer, "pickle": pickle_importer}


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


def _as_dict(model: BaseModel):
    field_keys = model.model_fields.keys()
    return {key: model.__dict__[key] for key in field_keys}
