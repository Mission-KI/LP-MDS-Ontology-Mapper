from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import List

from pytest import fixture, mark, raises

from edp import Service
from edp.types import (
    Augmentation,
    AugmentedColumn,
    Config,
    DataSetType,
    DataSpace,
    ExtendedDatasetProfile,
    License,
    Publisher,
    TemporalConsistency,
    UserProvidedEdpData,
)

DIR = Path(__file__).parent
ENCODING = "utf-8"
CSV_PATH: Path = DIR / "data/test.csv"
CSV_HEADERLESS_PATH = DIR / "data/test_headerless.csv"
XLSX_PATH: Path = DIR / "data/test.xlsx"
XLS_PATH: Path = DIR / "data/test.xls"
PICKLE_PATH = DIR / "data/test.pickle"
ZIP_PATH = DIR / "data/test.zip"


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


@mark.asyncio
async def test_load_unknown_dir(output_context, config_data):
    service = Service()
    with raises(FileNotFoundError):
        await service._compute_asset(Path("/does/not/exist/"), config_data, output_context)


async def test_analyse_pickle(output_context, config_data):
    service = Service()
    computed_data = await service._compute_asset(PICKLE_PATH, config_data, output_context)
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
async def test_analyse_csv(output_context, config_data):
    service = Service()
    json_file = await service.analyse_asset(CSV_PATH, config_data, output_context)
    edp: ExtendedDatasetProfile = read_edp(output_context.build_full_path(json_file))
    assert edp.assetId == config_data.userProvidedEdpData.assetId
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
async def test_analyse_csv_no_headers(output_context, user_provided_data):
    config_data = Config(userProvidedEdpData=user_provided_data)
    service = Service()
    json_file = await service.analyse_asset(CSV_HEADERLESS_PATH, config_data, output_context)
    edp: ExtendedDatasetProfile = read_edp(output_context.build_full_path(json_file))
    assert edp.assetId == config_data.userProvidedEdpData.assetId
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
async def test_analyse_xlsx(output_context, config_data):
    service = Service()
    json_file = await service.analyse_asset(XLSX_PATH, config_data, output_context)
    edp: ExtendedDatasetProfile = read_edp(output_context.build_full_path(json_file))
    assert edp.assetId == config_data.userProvidedEdpData.assetId
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
async def test_analyse_xls(output_context, config_data):
    service = Service()
    json_file = await service.analyse_asset(XLS_PATH, config_data, output_context)
    edp: ExtendedDatasetProfile = read_edp(output_context.build_full_path(json_file))
    assert edp.assetId == config_data.userProvidedEdpData.assetId
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
async def test_analyse_zip(output_context, config_data):
    service = Service()
    json_file = await service.analyse_asset(ZIP_PATH, config_data, output_context)
    edp = read_edp(output_context.build_full_path(json_file))
    assert edp.assetId == config_data.userProvidedEdpData.assetId
    assert "zip" in edp.compression.algorithms
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50


@mark.asyncio
async def test_raise_on_only_unknown_datasets(tmp_path, output_context, config_data):
    service = Service()
    file_path = tmp_path / "unsupported.txt"
    with open(file_path, "wt", encoding="utf-8") as file:
        file.write("This type is not supported")
    with raises((RuntimeWarning, RuntimeError)):
        await service.analyse_asset(tmp_path, config_data, output_context)


def read_edp(json_file: PurePosixPath):
    with open(json_file, "r") as file:
        json_data = file.read()
    return ExtendedDatasetProfile.model_validate_json(json_data)


@mark.asyncio
async def test_analyse_csv_daseen_context(daseen_output_context, config_data):
    service = Service()
    await service.analyse_asset(CSV_PATH, config_data, daseen_output_context)


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
