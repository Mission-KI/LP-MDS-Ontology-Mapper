import json
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class DataSpace(BaseModel):
    dataSpaceId: int = Field(
        strict=True, description=" Identifier that describes the data space in which the asset can be found"
    )
    name: str = Field(strict=True, description="Name of the dataspace")
    url: str = Field(strict=True, description="URL of the dataspace")


class DataSetVolume(str, Enum):
    kb = "KB"
    mb = "MB"
    gb = "GB"
    tb = "TB"
    pb = "PB"


class DataSetFrequenzy(str, Enum):
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


class DataSetStructure(str, Enum):
    structured = "structured (CSV)"
    semi_structured = "semi-structured (xml/json)"
    image = "image"
    video = "video"
    audio = "audio"
    documents = "documents"


class NumericColumnTypes(str, Enum):
    integer = "int"
    floating_point = "float"


class DataSetCompression(str, Enum):
    none = "None"
    gzip = "gzip"
    zip = "zip"
    tar = "tar.gz"
    seven_zip = "7zip"


class TemporalConsistencyFrequency(str, Enum):
    second = "s"
    minute = "min"
    hour = "h"
    day = "d"


class BaseColumn(BaseModel):
    name: str = Field(strict=True, description="Name of the column")


class NumericColumn(BaseColumn):
    min: Optional[float] = Field(default=None, strict=True)
    max: Optional[float] = Field(default=None, strict=True)
    mean: Optional[float] = Field(default=None, strict=True)
    median: Optional[float] = Field(default=None, strict=True)
    stddev: Optional[float] = Field(default=None, strict=True)
    data_type: NumericColumnTypes = Field(strict=True)


class TemporalConsistency(BaseModel):
    frequency: TemporalConsistencyFrequency = Field(strict=True)
    stable: bool = Field(strict=True)
    differentAbundancies: int = Field(strict=True)


class DateTimeColumnDefinition(BaseColumn):
    granularity: Optional[int] = Field(strict=True)
    temporalConsistency: List[TemporalConsistency] = Field(strict=True)


class StructuredEDPDataSet(BaseModel):
    columnCount: int = Field(
        strict=True,
        description="Number of columns",
    )
    rowCount: int = Field(
        strict=True,
        description="Number of row",
    )

    columns: List[Union[NumericColumn, DateTimeColumnDefinition]] = Field(
        strict=True, description="The dataset's columns"
    )


EDP_DATASET_SPECIALIZATION = Union[StructuredEDPDataSet]


class Schema(BaseModel):
    assetId: int = Field(strict=True, description="The asset ID is a unique identifier for an asset within a data room")
    assetName: str = Field(strict=True, description="Name of the asset")
    assetUrl: str = Field(
        strict=True, description="The URL via which the asset can be found in the published data room"
    )
    assetDataCategory: str = Field(
        default=None,
        strict=True,
        description="A data room-specific categorization of the asset (e.g. https://github.com/Mobility-Data-Space/mobility-data-space/wiki/MDS-Ontology",
    )
    dataSpace: DataSpace = Field(strict=True, description="Dataspace the asset can be found")
    publisherId: int = Field(
        strict=True, description="Identifier that describes the data provider that placed the asset in the data room"
    )
    licenseId: int = Field(
        strict=True,
        description="Identifier, which describes the data license under which the asset is made available by the data provider (see also https://www.dcat-ap.de/def/licenses/)",
    )
    assetVolume: int = Field(strict=True, description="Volume of the uncompressed asset in MB")
    assetVolumeCompressed: int = Field(strict=True, description="Volume of the compressed asset in MB")
    assetCompressionAlgorithm: DataSetCompression = Field(strict=True, description="Compression of the dataset")

    assetDescription: Optional[str] = Field(default=None, strict=True, description="Description of the asset")
    assetTags: Optional[List[str]] = Field(default=[], strict=True, description="Optional list of tags")
    assetDataSubCategory: Optional[str] = Field(
        default=None, strict=True, description="A data room-specific sub-categorization for assetDataCategory"
    )
    assetTransferTypeFlag: Optional[DataSetTransfer] = Field(
        default=None, strict=True, description="Describes whether an asset grows steadily over time "
    )
    assetImmutabilityFlag: Optional[DataSetImmutability] = Field(
        default=None, strict=True, description="Is the dataset immutable"
    )
    assetGrowthFlag: Optional[DataSetVolume] = Field(
        default=None,
        strict=True,
        description="If transfer is frequent, this parameter gives the growth rate of the dataset per day",
    )
    assetTransferTypeFlag: Optional[DataSetFrequenzy] = Field(
        default=None,
        strict=True,
        description="If transfer is frequent, this parameter gives the update frequency",
    )
    hasNdaFlag: Optional[bool] = Field(
        default=None,
        strict=True,
        description="Describes whether the use of the asset is subject to a non-disclosure agreement (“NDA”) between the data provider and the data user",
    )
    nda: Optional[str] = Field(default=None, strict=True, description="Identifier that describes or links to the NDA")
    assetStructure: DataSetStructure = Field(default=None, strict=True, description="Structure of the dataset")
    asset: EDP_DATASET_SPECIALIZATION = Field(
        default=[],
        strict=True,
        description="Additional columns dependent on the type of the dataset",
    )


def main():
    with open("edp_schema.json", "w") as f:
        json.dump(Schema.model_json_schema(), f)


if __name__ == "__main__":
    main()
