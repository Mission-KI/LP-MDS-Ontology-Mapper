from argparse import ArgumentParser
from datetime import datetime, timedelta
from enum import Enum
from importlib.metadata import version as get_version
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Optional, Set, Union

from pydantic import AnyUrl, BaseModel, Field, TypeAdapter


class DataSpace(BaseModel):
    dataSpaceId: int = Field(description=" Identifier that describes the data space in which the asset can be found")
    name: str = Field(description="Name of the dataspace")
    url: str = Field(description="URL of the dataspace")


class DataSetVolume(str, Enum):
    kb = "KB"
    mb = "MB"
    gb = "GB"
    tb = "TB"
    pb = "PB"


class DataSetFrequency(str, Enum):
    second = "second"
    minute = "minute"
    hour = "hour"
    day = "day"


class DataSetTransfer(str, Enum):
    static = "static"
    frequent = "inflationary"


class DataSetImmutability(str, Enum):
    immutable = "immutable"
    not_immutable = "not-immutable"


class DataSetType(str, Enum):
    structured = "structured"
    semi_structured = "semi-structured"
    image = "image"
    video = "video"
    audio = "audio"
    documents = "documents"


class DataSetCompression(str, Enum):
    none = "None"
    gzip = "gzip"
    zip = "zip"
    tar = "tar.gz"
    seven_zip = "7zip"


class TemporalConsistency(BaseModel):
    timeScale: timedelta
    stable: bool
    differentAbundancies: int
    abundances: List


Numeric = Union[int, float, timedelta, complex]

FileReference = Union[PurePosixPath, AnyUrl]
ImageList = List[FileReference]


class Gap(BaseModel):
    timeScale: timedelta = Field(description="Timescale for which gaps are given")
    numberOfGaps: int = Field(description="Number of gaps at the given timescale")


class Augmentation(BaseModel):
    sourceColumns: List[str] = Field(description="List of source columns on which the augmented column is based")
    formula: Optional[str] = Field(
        default=None,
        description="The calculation that was applied to the source columns to create the augmented column",
    )
    parameters: List[str] = Field(default_factory=list, description="The parameters used for the calculation")


class AugmentedColumn(BaseModel):
    name: str = Field(description="Name of the augmented column")
    file: Optional[PurePosixPath] = Field(
        default=None,
        description="Path of the augmented file this column was added. If augmentedFile is None, EDPS will assume that the augmented column contained in all files.",
    )
    augmentation: Augmentation = Field(description="Augmentation information")


class _BaseColumn(BaseModel):
    name: str = Field(description="Name of the column")
    nonNullCount: int = Field(description="Number of non empty entries in the column")
    nullCount: int = Field(description="Number of empty entries in the column")
    numberUnique: int = Field(description="Number of unique values")
    augmentation: Optional[Augmentation] = Field(
        default=None, description="If this column was augmented this filed contains all releveant information"
    )


class TimeBasedGraph(BaseModel):
    timeBaseColumn: str
    file: FileReference


class NumericColumn(_BaseColumn):
    min: Numeric
    max: Numeric
    mean: Numeric
    median: Numeric
    stddev: Numeric
    upperPercentile: Numeric = Field(description="Value of the upper 1% quantile")
    lowerPercentile: Numeric = Field(description="Value of the lower 1% quantile")
    upperQuantile: Numeric = Field(description="Value of the upper 25% quantile")
    lowerQuantile: Numeric = Field(description="Value of the lower 25% quantile")
    percentileOutlierCount: int = Field(description="Number of elements in the lower or upper percentile")
    upperZScore: Numeric = Field(description="Value of the upper standard score")
    lowerZScore: Numeric = Field(description="Value of the lower standard score")
    zScoreOutlierCount: int = Field(description="Number of elements outside the lower and upper standard scores")
    upperIQR: Numeric = Field(description="Value of the upper limit of the inter quartile range (25%)")
    lowerIQR: Numeric = Field(description="Value of the lower limit of the inter quartile range (25%)")
    iqr: Numeric = Field(description="Value of the inter quartile range")
    iqrOutlierCount: int = Field(description="Number of elements outside of the inter quartile range")
    distribution: str = Field(description="The best fitting distribution for the data in this column")
    distributionGraph: Optional[FileReference] = Field(
        default=None, description="Link to the combined histogram/distribution graph"
    )
    boxPlot: FileReference = Field(description="Link to the box plot of this column")
    seasonalities: List[TimeBasedGraph] = Field(
        default_factory=list, description="Seasonality graphs oer all available date time columns"
    )
    trends: List[TimeBasedGraph] = Field(
        default_factory=list, description="Trend graphs over all available date time columns"
    )
    residuals: List[TimeBasedGraph] = Field(
        default_factory=list, description="Residual graphs over all available date time columns"
    )
    weights: List[TimeBasedGraph] = Field(
        default_factory=list, description="Weights graphs over all available date time columns"
    )
    dataType: str


class TemporalCover(BaseModel):
    earliest: datetime
    latest: datetime


class DateTimeColumn(_BaseColumn):
    temporalCover: TemporalCover
    all_entries_are_unique: bool
    monotonically_increasing: bool
    monotonically_decreasing: bool
    granularity: Optional[int] = Field(default=None)
    temporalConsistencies: List[TemporalConsistency] = Field(description="Temporal consistency at given timescale")
    gaps: List[Gap] = Field(description="Number of gaps at given timescale")


class StringColumn(_BaseColumn):
    pass


class StructuredDataSet(BaseModel):
    name: PurePosixPath = Field(description="Name of the structured dataset")
    rowCount: int = Field(
        description="Number of row",
    )
    columnCount: int = Field(
        description="Number of columns",
    )
    numericColumnCount: int = Field("Numeric column count")
    datetimeColumnCount: int = Field("Datetime column count")
    stringColumnCount: int = Field("String column count")

    correlationGraph: Optional[FileReference] = Field(
        default=None, description="Reference to a correlation graph of the data columns"
    )
    numericColumns: List[NumericColumn] = Field(description="Numeric columns in this dataset")
    datetimeColumns: List[DateTimeColumn] = Field(description="Datetime columns in this dataset")
    stringColumns: List[StringColumn] = Field(
        description="Columns that could only be interpreted as string by the analysis"
    )

    @property
    def all_columns(self) -> Iterator[_BaseColumn]:
        yield from self.numericColumns
        yield from self.datetimeColumns
        yield from self.stringColumns

    def get_columns_dict(self) -> Dict[str, _BaseColumn]:
        return {column.name: column for column in self.all_columns}


class Publisher(BaseModel):
    id: str = Field(description="Unique identifier of the publisher")
    name: str = Field(description="Name of the publisher")
    url: Optional[str] = Field(default=None, description="URL to the publisher")


class License(BaseModel):
    name: str = Field(description="Name of the license")
    url: Optional[str] = Field(default=None, description="URL describing the license")


class UserProvidedEdpData(BaseModel):
    """The part of the EDP dataset that can not be automatically generated, but needs to be provided by the user."""

    assetId: str = Field(description="The asset ID is a unique identifier for an asset within a data room")
    name: str = Field(description="Name of the asset")
    url: str = Field(description="The URL via which the asset can be found in the published data room")
    dataCategory: str = Field(
        default=None,
        description="A data room-specific categorization of the asset (e.g. https://github.com/Mobility-Data-Space/mobility-data-space/wiki/MDS-Ontology",
    )
    dataSpace: DataSpace = Field(description="Dataspace the asset can be found")
    publisher: Publisher = Field(description="Provider that placed the asset in the data room")
    publishDate: datetime = Field(description="Date on which this asset has been published")
    license: License = Field(
        description="Describes the data license under which the asset is made available by the data provider (see also https://www.dcat-ap.de/def/licenses/)"
    )
    description: Optional[str] = Field(default=None, description="Description of the asset")
    tags: Optional[List[str]] = Field(default_factory=list, description="Optional list of tags")
    dataSubCategory: Optional[str] = Field(
        default=None, description="A data room-specific sub-categorization for assetDataCategory"
    )
    version: Optional[str] = Field(default=None, description="Provide supplied version of the asset")
    transferTypeFlag: Optional[DataSetTransfer] = Field(
        default=None, description="Describes whether an asset grows steadily over time "
    )
    immutabilityFlag: Optional[DataSetImmutability] = Field(default=None, description="Is the dataset immutable")
    growthFlag: Optional[DataSetVolume] = Field(
        default=None,
        description="If transfer is frequent, this parameter gives the growth rate of the dataset per day",
    )
    transferTypeFrequency: Optional[DataSetFrequency] = Field(
        default=None,
        description="If transfer is frequent, this parameter gives the update frequency",
    )
    nda: Optional[str] = Field(
        default=None, description="Identifier that describes or links to the non disclosure agreement"
    )
    dpa: Optional[str] = Field(default=None, description="Identifier that describes or links a dpa")
    dataLog: Optional[str] = Field(default=None, description="Description or links to data log")


class Config(BaseModel):
    userProvidedEdpData: UserProvidedEdpData = Field(description="User provided EDP meta data")
    augmentedColumns: List[AugmentedColumn] = Field(default_factory=list, description="List of augmented columns")


class Compression(BaseModel):
    algorithms: Set[str] = Field(description="List of all used compression algorithms in an asset")
    extractedSize: int = Field(description="Size in bytes when all compressions got extracted")


class ComputedEdpData(BaseModel):
    edps_version: str = Field(default=get_version("edp"), description="Version of EDPS used to generate this EDP")
    volume: int = Field(description="Volume of the asset in MB")
    compression: Optional[Compression] = Field(default=None, description="Description of compressions used")
    dataTypes: Set[DataSetType] = Field(description="Types of data contained in this asset")
    temporalCover: Optional[TemporalCover] = Field(
        default=None, description="Earliest and latest dates contained in this asset"
    )

    structuredDatasets: List[StructuredDataSet] = Field(
        default_factory=list,
        description="Metadata for all datasets (files) detected to be structured (tables)",
    )


class ExtendedDatasetProfile(UserProvidedEdpData, ComputedEdpData):
    pass


def export_edp_schema():
    args = _get_args()
    output: Path = args.output
    if output.is_dir():
        output /= "edsp_schema.json"
    adapter = TypeAdapter(Dict[str, Any])
    with open(output, "wb") as file:
        file.write(adapter.dump_json(ExtendedDatasetProfile.model_json_schema()))


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, help="PurePosixPath to output the schema to")
    return parser.parse_args()
