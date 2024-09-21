import json
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class DataSpace(BaseModel):
    name: str = (Field(strict=True, description="Name of the dataspace"),)
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
    frequent = "frequent"


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


class BaseColumn(BaseModel):
    name: str = Field(strict=True, description="Name of the column")


class NumericColumn(BaseColumn):
    min: Optional[float] = Field(default=None, strict=True)
    max: Optional[float] = Field(default=None, strict=True)
    mean: Optional[float] = Field(default=None, strict=True)
    median: Optional[float] = Field(default=None, strict=True)
    stddev: Optional[float] = Field(default=None, strict=True)
    data_type: NumericColumnTypes = Field(strict=True)


class TimeFieldColumn(BaseColumn):
    granularity: Optional[int] = Field(strict=True)


EDP_COLUMN = Union[NumericColumn, TimeFieldColumn]


class EDPSchema(BaseModel):
    data_space: DataSpace = Field(strict=True)
    volume: DataSetVolume = Field(strict=True, description="Volume of the dataset")
    transfer: DataSetTransfer = Field(
        strict=True, description="Is the dataset frequently updated, or static"
    )
    frequency: DataSetFrequenzy = Field(
        strict=True,
        description="If transfer is frequent, this parameter gives the update frequency",
    )
    growth_rate: Optional[DataSetVolume] = Field(
        default=None,
        strict=True,
        description="If transfer is frequent, this parameter gives the growth rate of the dataset per day",
    )
    immutability: DataSetImmutability = Field(
        strict=True, description="Is the dataset immutable"
    )
    structure: DataSetStructure = Field(
        strict=True, description="Structure of the dataset"
    )
    compression: DataSetCompression = Field(
        strict=True, description="Compression of the dataset"
    )
    column_count: int = Field(
        strict=True,
        description="Number of columns",
    )
    row_count: int = Field(
        strict=True,
        description="Number of row",
    )
    columns: List[EDP_COLUMN] = Field(strict=True, description="The dataset's columns")


with open("edp_schema.json", "w") as f:
    json.dump(EDPSchema.model_json_schema(), f)
