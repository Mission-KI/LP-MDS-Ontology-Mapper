from logging import getLogger
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List, Optional
from warnings import warn

from extended_dataset_profile.models.v0.edp import (
    ArchiveDataSet,
    DataSetType,
    DocumentDataSet,
    ExtendedDatasetProfile,
    FileReference,
    ImageDataSet,
    StructuredDataSet,
    TemporalCover,
    UnstructuredTextDataSet,
    VideoDataSet,
    _BaseColumn,
)
from pandas import DataFrame
from pydantic import BaseModel

from edps.analyzers.pandas import determine_periodicity
from edps.compression import DECOMPRESSION_ALGORITHMS
from edps.file import calculate_size
from edps.filewriter import write_edp
from edps.importers import get_importable_types
from edps.taskcontext import TaskContext
from edps.types import AugmentedColumn, ComputedEdpData, Config, DataSet


class UserInputError(RuntimeError):
    pass


class Service:
    def __init__(self):
        _logger = getLogger(__name__)
        _logger.info("Initializing")

        _logger.info("The following data types are supported: [%s]", get_importable_types())
        implemented_decompressions = [key for key, value in DECOMPRESSION_ALGORITHMS.items() if value is not None]
        _logger.info("The following compressions are supported: [%s]", ", ".join(implemented_decompressions))

    async def analyse_asset(self, ctx: TaskContext, config: Config) -> FileReference:
        """Let the service analyse the assets in ctx.input_path

        Parameters
        ----------
        ctx : TaskContext
            Gives access to the appropriate logger and output_context and allows executing sub-tasks.
        config_data : Config
            The meta and config information about the asset supplied by the data space. These can not get calculated and must be supplied
            by the user.

        Returns
        -------
        FileReference
            File path or URL to the generated edp.service.
        """
        computed_data = await self._compute_asset(ctx, config)
        user_data = config.userProvidedEdpData
        edp = ExtendedDatasetProfile(**_as_dict(computed_data), **_as_dict(user_data))
        json_name = user_data.assetId + ("_" + user_data.version if user_data.version else "")
        return await write_edp(ctx, PurePosixPath(json_name), edp)

    async def _compute_asset(self, ctx: TaskContext, config: Config) -> ComputedEdpData:
        input_files = [path for path in ctx.input_path.iterdir() if path.is_file()]
        number_input_files = len(input_files)
        if number_input_files != 1:
            raise UserInputError(f"Expected exactly one input file in input_path. Got {number_input_files}.")
        file = input_files[0]

        await ctx.import_file(file, file.name)
        datasets = list(ctx.collect_datasets())

        if len(datasets) == 0:
            raise RuntimeError("Was not able to analyze any datasets in this asset")
        number_top_level_datasets = sum(1 for dataset in datasets if dataset.parentUuid is None)
        if number_top_level_datasets != 1:
            raise UserInputError(
                f"There must be exactly one top level file/archive! Found {number_top_level_datasets}."
            )

        computed_edp_data = self._create_computed_edp_data(file, datasets)
        computed_edp_data = await self._add_augmentation(ctx, config, computed_edp_data)
        if self._has_temporal_columns(computed_edp_data):
            computed_edp_data.temporalCover = self._get_overall_temporal_cover(computed_edp_data)
        computed_edp_data.periodicity = self._get_overall_temporal_consistency(computed_edp_data)
        return computed_edp_data

    def _create_computed_edp_data(self, path: Path, datasets: List[DataSet]) -> ComputedEdpData:
        edp = ComputedEdpData(volume=calculate_size(path))
        for dataset in datasets:
            if isinstance(dataset, ArchiveDataSet):
                edp.archiveDatasets.append(dataset)
                edp.dataTypes.add(DataSetType.archive)
            elif isinstance(dataset, StructuredDataSet):
                edp.structuredDatasets.append(dataset)
                edp.dataTypes.add(DataSetType.structured)
            elif isinstance(dataset, UnstructuredTextDataSet):
                edp.unstructuredTextDatasets.append(dataset)
                edp.dataTypes.add(DataSetType.unstructuredText)
            elif isinstance(dataset, ImageDataSet):
                edp.imageDatasets.append(dataset)
                edp.dataTypes.add(DataSetType.image)
            elif isinstance(dataset, VideoDataSet):
                edp.videoDatasets.append(dataset)
                edp.dataTypes.add(DataSetType.video)
            elif isinstance(dataset, DocumentDataSet):
                edp.documentDatasets.append(dataset)
                edp.dataTypes.add(DataSetType.documents)
            else:
                raise NotImplementedError(f'Did not expect dataset type "{type(dataset)}"')
        return edp

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
                message = (
                    f'Augmented column "{augmented_column.name}" is not known in file "{augmented_column.datasetName}'
                )
                ctx.logger.warning(message)
                warn(message)

        for augmented_column in config_data.augmentedColumns:
            if augmented_column.datasetName is None:
                augment_column_in_all_files(augmented_column)
            else:
                try:
                    dataset = structured_datasets[augmented_column.datasetName]
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
        all_temporal_consistencies = list(self._iterate_all_temporal_consistencies(edp))
        if len(all_temporal_consistencies) == 0:
            return None
        sum_temporal_consistencies = sum(all_temporal_consistencies[1:], all_temporal_consistencies[0])
        return determine_periodicity(
            sum_temporal_consistencies["numberOfGaps"], sum_temporal_consistencies["differentAbundancies"]
        )

    def _iterate_all_temporal_consistencies(self, edp: ComputedEdpData) -> Iterator[DataFrame]:
        for dataset in edp.structuredDatasets:
            for row in dataset.datetimeColumns:
                dataframe = DataFrame(
                    index=[item.timeScale for item in row.temporalConsistencies],
                )
                dataframe["differentAbundancies"] = [item.differentAbundancies for item in row.temporalConsistencies]
                dataframe["numberOfGaps"] = [item.numberOfGaps for item in row.temporalConsistencies]
                yield dataframe


def _as_dict(model: BaseModel):
    field_keys = model.model_fields.keys()
    return {key: model.__dict__[key] for key in field_keys}
