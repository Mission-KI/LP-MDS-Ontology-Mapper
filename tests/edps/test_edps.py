import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, List

from pytest import fixture, mark, raises

from edps import Service
from edps.task import TaskContext
from edps.types import (
    Augmentation,
    AugmentedColumn,
    ComputedEdpData,
    Config,
    DataSetType,
    DataSpace,
    ExtendedDatasetProfile,
    FileReference,
    License,
    Publisher,
    TemporalConsistency,
    UserProvidedEdpData,
)

ENCODING = "utf-8"


@fixture
def user_provided_data():
    return UserProvidedEdpData(
        assetId="my-dataset-id",
        name="dataset-dummy-name",
        url="https://beebucket.ai/en/",
        dataCategory="TestDataCategory",
        dataSpace=DataSpace(name="TestDataSpace", url="https://beebucket.ai/en/"),
        publisher=Publisher(name="beebucket"),
        license=License(url="https://opensource.org/license/mit"),
        description="Our very first test edp",
        publishDate=datetime(year=1995, month=10, day=10, hour=10, tzinfo=timezone.utc),
        version="2.3.1",
        tags=["test", "csv"],
        freely_available=True,
    )


@fixture
def config_data(user_provided_data):
    augmented_columns = [
        AugmentedColumn(
            name="aufenthalt",
            augmentation=Augmentation(sourceColumns=["einfahrt", "ausfahrt"]),
        )
    ]
    return Config(userProvidedEdpData=user_provided_data, augmentedColumns=augmented_columns)


@fixture
def analyse_asset_fn(ctx, config_data, output_context) -> Callable[[Path], Awaitable[FileReference]]:
    return lambda path: analyse_asset(ctx, config_data, path)


@fixture
def compute_asset_fn(ctx, config_data, output_context) -> Callable[[Path], Awaitable[ComputedEdpData]]:
    return lambda path: compute_asset(ctx, config_data, path)


@mark.asyncio
async def test_load_unknown_dir(analyse_asset_fn):
    with raises(FileNotFoundError):
        await analyse_asset_fn(Path("/does/not/exist/"))


async def test_analyse_pickle(path_data_test_pickle, compute_asset_fn, config_data):
    computed_data = await compute_asset_fn(path_data_test_pickle)

    assert len(computed_data.structuredDatasets) == 1
    assert computed_data.periodicity == "hours"

    dataset = computed_data.structuredDatasets[0]
    assert len(dataset.datetimeColumns) == 2
    assert len(dataset.numericColumns) == 2
    assert len(dataset.stringColumns) == 1

    aufenthalt = next(item for item in dataset.numericColumns if item.name == "aufenthalt")
    assert aufenthalt.dataType == "UInt32"
    assert aufenthalt.augmentation == config_data.augmentedColumns[0].augmentation

    parkhaus = next(item for item in dataset.numericColumns if item.name == "parkhaus")
    assert parkhaus.dataType == "UInt8"

    einfahrt = next(item for item in dataset.datetimeColumns if item.name == "einfahrt")
    assert einfahrt.temporalCover.earliest == datetime.fromisoformat("2016-01-01 00:03:14")
    assert einfahrt.temporalCover.latest == datetime.fromisoformat("2016-01-01 11:50:45")
    _assert_pickle_temporal_consistencies(einfahrt.temporalConsistencies)
    assert einfahrt.periodicity == "hours"

    assert len(computed_data.dataTypes) == 1
    assert DataSetType.structured in computed_data.dataTypes

    assert computed_data.temporalCover is not None
    assert computed_data.temporalCover.earliest == datetime.fromisoformat("2016-01-01 00:03:14")
    assert computed_data.temporalCover.latest == datetime.fromisoformat("2016-01-01 21:13:08")


@mark.asyncio
async def test_analyse_csv(path_data_test_csv, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_csv)
    assert edp.compression is None
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50
    assert edp.structuredDatasets[0].stringColumns[0].name == "uuid"
    assert edp.structuredDatasets[0].datetimeColumns[0].name == "einfahrt"
    assert edp.structuredDatasets[0].datetimeColumns[0].format == "ISO8601"
    assert edp.structuredDatasets[0].datetimeColumns[1].name == "ausfahrt"
    assert edp.structuredDatasets[0].datetimeColumns[1].format == "ISO8601"
    assert edp.structuredDatasets[0].numericColumns[0].name == "aufenthalt"
    assert edp.structuredDatasets[0].numericColumns[0].dataType == "UInt32"
    assert edp.structuredDatasets[0].numericColumns[1].name == "parkhaus"
    assert edp.structuredDatasets[0].numericColumns[1].dataType == "UInt8"


@mark.asyncio
async def test_analyse_roundtrip_csv(path_data_test_csv, analyse_asset_fn, output_context, config_data):
    edp_file = await analyse_asset_fn(path_data_test_csv)
    edp_file_path = output_context.build_full_path(edp_file)
    edp = read_edp_file(edp_file_path)
    assert edp.assetId == config_data.userProvidedEdpData.assetId
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50


@mark.asyncio
async def test_analyse_csv_no_headers(path_data_test_headerless_csv, ctx, user_provided_data, output_context):
    # We can't use the default config.
    config_data = Config(userProvidedEdpData=user_provided_data)
    edp = await compute_asset(ctx, config_data, path_data_test_headerless_csv)

    assert edp.compression is None
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50
    assert edp.structuredDatasets[0].stringColumns[0].name == "col000"
    assert edp.structuredDatasets[0].datetimeColumns[0].name == "col001"
    assert edp.structuredDatasets[0].datetimeColumns[0].format == "ISO8601"
    assert edp.structuredDatasets[0].datetimeColumns[1].name == "col002"
    assert edp.structuredDatasets[0].datetimeColumns[1].format == "ISO8601"
    assert edp.structuredDatasets[0].numericColumns[0].name == "col003"
    assert edp.structuredDatasets[0].numericColumns[0].dataType == "UInt32"
    assert edp.structuredDatasets[0].numericColumns[1].name == "col004"
    assert edp.structuredDatasets[0].numericColumns[1].dataType == "UInt8"


@mark.asyncio
async def test_analyse_xlsx(path_data_test_xlsx, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_xlsx)
    assert edp.compression is None
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50
    assert edp.structuredDatasets[0].stringColumns[0].name == "uuid"
    assert edp.structuredDatasets[0].datetimeColumns[0].name == "einfahrt"
    assert edp.structuredDatasets[0].datetimeColumns[0].format == "NATIVE"
    assert edp.structuredDatasets[0].datetimeColumns[1].name == "ausfahrt"
    assert edp.structuredDatasets[0].datetimeColumns[1].format == "NATIVE"
    assert edp.structuredDatasets[0].numericColumns[0].name == "aufenthalt"
    assert edp.structuredDatasets[0].numericColumns[0].dataType == "UInt32"
    assert edp.structuredDatasets[0].numericColumns[1].name == "parkhaus"
    assert edp.structuredDatasets[0].numericColumns[1].dataType == "UInt8"


@mark.asyncio
async def test_analyse_xls(path_data_test_xls, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_xls)
    assert edp.compression is None
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50
    assert edp.structuredDatasets[0].stringColumns[0].name == "uuid"
    assert edp.structuredDatasets[0].datetimeColumns[0].name == "einfahrt"
    assert edp.structuredDatasets[0].datetimeColumns[0].format == "NATIVE"
    assert edp.structuredDatasets[0].datetimeColumns[1].name == "ausfahrt"
    assert edp.structuredDatasets[0].datetimeColumns[1].format == "NATIVE"
    assert edp.structuredDatasets[0].numericColumns[0].name == "aufenthalt"
    assert edp.structuredDatasets[0].numericColumns[0].dataType == "UInt32"
    assert edp.structuredDatasets[0].numericColumns[1].name == "parkhaus"
    assert edp.structuredDatasets[0].numericColumns[1].dataType == "UInt8"


@mark.asyncio
async def test_analyse_zip(path_data_test_zip, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_zip)
    assert "zip" in edp.compression.algorithms
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50


@mark.asyncio
async def test_analyse_multiassets_zip(path_data_test_multiassets_zip, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_multiassets_zip)
    assert "zip" in edp.compression.algorithms
    assert len(edp.structuredDatasets) == 4
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50
    assert {str(dataset.name) for dataset in edp.structuredDatasets} == {
        "test_multiassets_zip/csv/test.csv",
        "test_multiassets_zip/xls/test.xls",
        "test_multiassets_zip/xlsx/test.xlsx",
        "test_multiassets_zip/zip/test_zip/test.csv",
    }


@mark.asyncio
async def test_raise_on_only_unknown_datasets(analyse_asset_fn, tmp_path):
    file_path = tmp_path / "unsupported.txt"
    with open(file_path, "wt", encoding="utf-8") as file:
        file.write("This type is not supported")
    with raises((RuntimeWarning, RuntimeError)):
        await analyse_asset_fn(file_path)


@mark.asyncio
async def test_analyse_csv_daseen_context(path_data_test_csv, ctx, daseen_output_context, config_data):
    await analyse_asset(ctx, config_data, path_data_test_csv)


async def analyse_asset(ctx: TaskContext, config_data: Config, asset_path: Path):
    _copy_asset_to_working_dir(asset_path, ctx)
    return await Service().analyse_asset(ctx, config_data)


async def compute_asset(ctx: TaskContext, config_data: Config, asset_path: Path):
    _copy_asset_to_working_dir(asset_path, ctx)
    return await Service()._compute_asset(ctx, config_data)


def _copy_asset_to_working_dir(asset_path: Path, ctx: TaskContext):
    shutil.rmtree(ctx.input_path, ignore_errors=True)
    ctx.input_path.mkdir()
    shutil.copy(asset_path, ctx.input_path / asset_path.name)


def _assert_pickle_temporal_consistencies(
    temporal_consistencies: List[TemporalConsistency],
):
    microseconds_consistency = temporal_consistencies[0]
    assert microseconds_consistency.timeScale == "microseconds"
    assert microseconds_consistency.stable is False
    assert microseconds_consistency.differentAbundancies == 50
    assert microseconds_consistency.numberOfGaps == 49

    milliseconds_consistency = temporal_consistencies[1]
    assert milliseconds_consistency.timeScale == "milliseconds"
    assert milliseconds_consistency.stable is False
    assert milliseconds_consistency.differentAbundancies == 50
    assert milliseconds_consistency.numberOfGaps == 49

    seconds_consistency = temporal_consistencies[2]
    assert seconds_consistency.timeScale == "seconds"
    assert seconds_consistency.stable is False
    assert seconds_consistency.differentAbundancies == 50
    assert seconds_consistency.numberOfGaps == 48

    minutes_consistency = temporal_consistencies[3]
    assert minutes_consistency.timeScale == "minutes"
    assert minutes_consistency.stable is False
    assert minutes_consistency.differentAbundancies == 47
    assert minutes_consistency.numberOfGaps == 44

    hours_consistency = temporal_consistencies[4]
    assert hours_consistency.timeScale == "hours"
    assert hours_consistency.stable is False
    assert hours_consistency.differentAbundancies == 12
    assert hours_consistency.numberOfGaps == 3

    days_consistency = temporal_consistencies[5]
    assert days_consistency.timeScale == "days"
    assert days_consistency.stable is True
    assert days_consistency.differentAbundancies == 1
    assert days_consistency.numberOfGaps == 0

    weeks_consistency = temporal_consistencies[6]
    assert weeks_consistency.timeScale == "weeks"
    assert weeks_consistency.stable is True
    assert weeks_consistency.differentAbundancies == 1
    assert weeks_consistency.numberOfGaps == 0

    month_consistency = temporal_consistencies[7]
    assert month_consistency.timeScale == "months"
    assert month_consistency.stable is True
    assert month_consistency.differentAbundancies == 1
    assert month_consistency.numberOfGaps == 0

    year_consistency = temporal_consistencies[8]
    assert year_consistency.timeScale == "years"
    assert year_consistency.stable is True
    assert year_consistency.differentAbundancies == 1
    assert year_consistency.numberOfGaps == 0


def read_edp_file(json_file: Path):
    with open(json_file, "r") as file:
        json_data = file.read()
    return ExtendedDatasetProfile.model_validate_json(json_data)
