from argparse import ArgumentParser
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, TypeAdapter


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
    interval: timedelta
    stable: bool
    differentAbundancies: int
    abundances: List


Numeric = Union[int, float, timedelta, complex]


class NumericColumn(BaseModel):
    min: Numeric
    max: Numeric
    mean: Numeric
    median: Numeric
    stddev: Numeric
    dataType: str


class DateTimeColumn(BaseModel):
    earliest: datetime
    latest: datetime
    all_entries_are_unique: bool
    monotonically_increasing: bool
    monotonically_decreasing: bool
    granularity: Optional[int] = Field(default=None)
    temporalConsistencies: List[TemporalConsistency]
    gaps: Dict[timedelta, int] = Field(description="Number of gaps at given timescale")


class StringColumn(BaseModel):
    pass


Column = Union[NumericColumn, DateTimeColumn, StringColumn]


class StructuredEDPDataSet(BaseModel):
    rowCount: int = Field(
        description="Number of row",
    )
    columns: Dict[str, Column] = Field(description="The dataset's columns")


Dataset = Union[StructuredEDPDataSet]


class Publisher(BaseModel):
    id: str = Field(description="Unique identifier of the publisher")
    name: str = Field(description="Name of the publisher")


class UserProvidedAssetData(BaseModel):
    """The part of the EDP dataset that can not be automatically generated, but needs to be provided by the user."""

    id: int = Field(description="The asset ID is a unique identifier for an asset within a data room")
    name: str = Field(description="Name of the asset")
    url: str = Field(description="The URL via which the asset can be found in the published data room")
    dataCategory: str = Field(
        default=None,
        description="A data room-specific categorization of the asset (e.g. https://github.com/Mobility-Data-Space/mobility-data-space/wiki/MDS-Ontology",
    )
    dataSpace: DataSpace = Field(description="Dataspace the asset can be found")
    publisher: Publisher = Field(description="Provider that placed the asset in the data room")
    licenseId: int = Field(
        description="Identifier, which describes the data license under which the asset is made available by the data provider (see also https://www.dcat-ap.de/def/licenses/)",
    )
    description: Optional[str] = Field(default=None, description="Description of the asset")
    tags: Optional[List[str]] = Field(default=[], description="Optional list of tags")
    dataSubCategory: Optional[str] = Field(
        default=None, description="A data room-specific sub-categorization for assetDataCategory"
    )
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


class Compression(BaseModel):
    algorithms: List[str] = Field(description="List of all used compression algorithms in an asset")
    extractedSize: int = Field(description="Size in bytes when all compressions got extracted")


class ComputedAssetData(BaseModel):
    volume: int = Field(description="Volume of the asset in MB")
    compression: Optional[Compression] = Field(default=None, description="Description of compressions used")
    dataTypes: Set[DataSetType] = Field(description="Types of data contained in this asset")

    datasets: List[Dataset] = Field(
        default_factory=list,
        description="Additional columns dependent on the type of the dataset",
    )


class Asset(UserProvidedAssetData, ComputedAssetData):
    pass


def export_edp_schema():
    args = _get_args()
    output: Path = args.output
    if output.is_dir():
        output /= "edsp_schema.json"
    adapter = TypeAdapter(Dict[str, Any])
    with open(output, "wb") as file:
        file.write(adapter.dump_json(Asset.model_json_schema()))


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, help="Path to output the schema to")
    return parser.parse_args()
