from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import AsyncIterator, Dict, Iterator, List, Optional, Set
from warnings import warn

from pydantic import BaseModel

from edps.compression import DECOMPRESSION_ALGORITHMS
from edps.file import File, calculate_size
from edps.importers import IMPORTERS
from edps.task import TaskContext
from edps.types import (
    AugmentedColumn,
    Compression,
    ComputedEdpData,
    Config,
    DataSet,
    DataSetType,
    ExtendedDatasetProfile,
    FileReference,
    StructuredDataSet,
    TemporalCover,
    _BaseColumn,
)


class Service:
    def __init__(self):
        _logger = getLogger(__name__)
        _logger.info("Initializing")

        _logger.info("The following data types are supported: [%s]", ", ".join(IMPORTERS))
        implemented_decompressions = [key for key, value in DECOMPRESSION_ALGORITHMS.items() if value is not None]
        _logger.info("The following compressions are supported: [%s]", ", ".join(implemented_decompressions))

    async def analyse_asset(self, ctx: TaskContext, config: Config, path: Path) -> FileReference:
        """Let the service analyse the given asset

        Parameters
        ----------
        ctx : TaskContext
            Gives access to the appriopriate logger and output_context and allows executing sub-tasks.
        path : Path
             Can be a single file or directory. If it is a directory, all files inside that directory will get analyzed.
        config_data : Config
            The meta and config information about the asset supplied by the data space. These can not get calculated and must be supplied
            by the user.

        Returns
        -------
        FileReference
            File path or URL to the generated edp.service.
        """
        computed_data = await self._compute_asset(ctx, config, path)
        user_data = config.userProvidedEdpData
        edp = ExtendedDatasetProfile(**_as_dict(computed_data), **_as_dict(user_data))
        json_name = user_data.assetId + ("_" + user_data.version if user_data.version else "")
        json_name = json_name.replace(".", "_")
        return await ctx.output_context.write_edp(json_name, edp)

    async def _compute_asset(self, ctx: TaskContext, config: Config, path: Path) -> ComputedEdpData:
        if not path.exists():
            raise FileNotFoundError(f'File "{path}" can not be found!')
        compression_algorithms: Set[str] = set()
        extracted_size = 0
        datasets: List[DataSet] = []
        base_path = path if path.is_dir() else path.parent

        async for child_file in self._walk_all_files(ctx, base_path, path, compression_algorithms):
            file_type = child_file.type
            extracted_size += child_file.size
            if file_type not in IMPORTERS:
                text = f'Import for "{file_type}" not yet supported'
                ctx.logger.warning(text)
                warn(text, RuntimeWarning)
                continue
            analyzer = await ctx.exec(IMPORTERS[file_type], child_file)
            async for dataset in ctx.exec(analyzer.analyze):
                datasets.append(dataset)

        compression: Optional[Compression]
        if len(compression_algorithms) == 0:
            compression = None
        else:
            compression = Compression(algorithms=compression_algorithms, extractedSize=extracted_size)

        if len(datasets) == 0:
            raise RuntimeError("Was not able to analyze any datasets in this asset")
        computed_edp_data = self._create_computed_edp_data(path, compression, datasets)
        computed_edp_data = await self._add_augmentation(ctx, config, computed_edp_data)
        if self._has_temporal_columns(computed_edp_data):
            computed_edp_data.temporalCover = self._get_overall_temporal_cover(computed_edp_data)
        computed_edp_data.periodicity = self._get_overall_temporal_consistency(computed_edp_data)
        return computed_edp_data

    def _create_computed_edp_data(
        self, path: Path, compression: Optional[Compression], datasets: List[DataSet]
    ) -> ComputedEdpData:
        structured_datasets: List[StructuredDataSet] = []
        data_types: Set[DataSetType] = set()

        for dataset in datasets:
            if isinstance(dataset, StructuredDataSet):
                structured_datasets.append(dataset)
                data_types.add(DataSetType.structured)
            else:
                raise NotImplementedError(f'Did not expect dataset type "{type(dataset)}"')

        return ComputedEdpData(
            volume=calculate_size(path),
            compression=compression,
            dataTypes=data_types,
            structuredDatasets=structured_datasets,
        )

    async def _walk_all_files(
        self, ctx: TaskContext, base_path: Path, path: Path, compressions: Set[str]
    ) -> AsyncIterator[File]:
        """Will yield all files, recursively through directories and archives."""

        if path.is_file():
            file = File(base_path, path)
            if file.type not in DECOMPRESSION_ALGORITHMS:
                yield file
            else:
                compressions.add(file.type)
                async with self._extract(file) as extracted_path:
                    async for child_file in self._walk_all_files(ctx, base_path, extracted_path, compressions):
                        yield child_file
        elif path.is_dir():
            for file_path in path.iterdir():
                async for child_file in self._walk_all_files(ctx, base_path, file_path, compressions):
                    yield child_file
        else:
            ctx.logger.warning('Can not extract or analyse "%s"', path)

    @asynccontextmanager
    async def _extract(self, file: File) -> AsyncIterator[Path]:
        archive_type = file.type
        if archive_type not in DECOMPRESSION_ALGORITHMS:
            raise RuntimeError(f'"{archive_type}" is not a know archive type')
        decompressor = DECOMPRESSION_ALGORITHMS[archive_type]
        if decompressor is None:
            raise NotImplementedError(f'Extracting "{archive_type}" is not implemented')
        archive_name = file.path.name
        directory = file.path.parent / archive_name.replace(".", "_")
        while directory.exists():
            directory /= "extracted"

        directory.mkdir()
        await decompressor.extract(file.path, directory)
        try:
            yield directory
        finally:
            rmtree(directory.absolute())

    async def _add_augmentation(self, ctx: TaskContext, config_data: Config, edp: ComputedEdpData) -> ComputedEdpData:
        structured_datasets = {dataset.name: dataset.get_columns_dict() for dataset in edp.structuredDatasets}

        def _get_all_matching_columns(name: str) -> Iterator[_BaseColumn]:
            for columns in structured_datasets.values():
                if name in columns:
                    yield columns[name]

        def augment_column_in_all_files(augmented_column: AugmentedColumn) -> None:
            columns = list(_get_all_matching_columns(augmented_column.name))
            if len(columns) == 0:
                message = f'No column "{augmented_column.name}" found in any dataset!'
                warn(message)
                ctx.logger.warning(message)
                return
            for column in columns:
                column.augmentation = augmented_column.augmentation

        def augment_column_in_file(augmented_column: AugmentedColumn, dataset: Dict[str, _BaseColumn]) -> None:
            try:
                dataset[augmented_column.name].augmentation = augmented_column.augmentation
            except KeyError:
                message = f'Augmented column "{augmented_column.name}" is not known in file "{augmented_column.file}'
                ctx.logger.warning(message)
                warn(message)

        for augmented_column in config_data.augmentedColumns:
            if augmented_column.file is None:
                augment_column_in_all_files(augmented_column)
            else:
                try:
                    dataset = structured_datasets[augmented_column.file]
                except KeyError:
                    message = f'"{augmented_column}" is not a known structured dataset!"'
                    warn(message)
                    ctx.logger.warning(message)
                    continue
                augment_column_in_file(augmented_column, dataset)

        return edp

    def _has_temporal_columns(self, edp: ComputedEdpData) -> bool:
        for structured in edp.structuredDatasets:
            if len(structured.datetimeColumns) > 0:
                return True

        return False

    def _get_overall_temporal_cover(self, edp: ComputedEdpData) -> TemporalCover:
        earliest = min(
            column.temporalCover.earliest
            for structured in edp.structuredDatasets
            for column in structured.datetimeColumns
        )
        latest = max(
            column.temporalCover.latest
            for structured in edp.structuredDatasets
            for column in structured.datetimeColumns
        )
        return TemporalCover(earliest=earliest, latest=latest)

    def _get_overall_temporal_consistency(self, edp: ComputedEdpData) -> Optional[str]:
        for dataset in edp.structuredDatasets:
            if dataset.primaryDatetimeColumn is not None:
                try:
                    column = next(
                        (column for column in dataset.datetimeColumns if column.name == dataset.primaryDatetimeColumn)
                    )
                except StopIteration:
                    continue
                if column.periodicity:
                    return column.periodicity
        return None


def _as_dict(model: BaseModel):
    field_keys = model.model_fields.keys()
    return {key: model.__dict__[key] for key in field_keys}
