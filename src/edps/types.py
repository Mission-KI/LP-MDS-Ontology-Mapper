import html
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, List, Optional, Set

from extended_dataset_profile import SchemaVersion
from extended_dataset_profile.models.v0.edp import (
    AssetProcessingStatus,
    Augmentation,
    Compression,
    DataSetFrequency,
    DataSetImmutability,
    DataSetTransfer,
    DataSetType,
    DataSetVolume,
    DataSpace,
    DocumentDataSet,
    ExtendedDatasetProfile,
    ImageDataSet,
    License,
    Publisher,
    StructuredDataSet,
    TemporalCover,
    UnstructuredTextDataSet,
    VideoDataSet,
)
from pydantic import BaseModel, Field, model_validator


def _get_edp_field(name: str) -> Any:
    """Function to avoid mypy from complaining about putting FieldInfo's into fields."""
    return ExtendedDatasetProfile.model_fields[name]


class UserProvidedEdpData(BaseModel):
    """
    The data provided by the caller of the EDP Service.

    This is a subset of the extended dataset profile.
    """

    assetId: str = _get_edp_field("assetId")
    name: str = _get_edp_field("name")
    url: str = _get_edp_field("url")
    dataCategory: str | None = _get_edp_field("dataCategory")
    dataSpace: DataSpace = _get_edp_field("dataSpace")
    publisher: Publisher = _get_edp_field("publisher")
    publishDate: datetime = _get_edp_field("publishDate")
    license: License = _get_edp_field("license")
    assetProcessingStatus: AssetProcessingStatus | None = _get_edp_field("assetProcessingStatus")
    description: str | None = _get_edp_field("description")
    tags: List[str] = _get_edp_field("tags")
    dataSubCategory: str | None = _get_edp_field("dataSubCategory")
    version: str | None = _get_edp_field("version")
    transferTypeFlag: DataSetTransfer | None = _get_edp_field("transferTypeFlag")
    immutabilityFlag: DataSetImmutability | None = _get_edp_field("immutabilityFlag")
    growthFlag: DataSetVolume | None = _get_edp_field("growthFlag")
    transferTypeFrequency: DataSetFrequency | None = _get_edp_field("transferTypeFrequency")
    nda: str | None = _get_edp_field("nda")
    dpa: str | None = _get_edp_field("dpa")
    dataLog: str | None = _get_edp_field("dataLog")
    freely_available: bool = _get_edp_field("freely_available")

    @model_validator(mode="before")
    def escape_all_string_fields(cls, data: Any) -> Any:
        return recursively_escape_strings(data)


class ComputedEdpData(BaseModel):
    """All fields of the extended dataset profile that get calculated by this service."""

    dataTypes: Set[DataSetType] = _get_edp_field("dataTypes")
    unstructuredTextDatasets: List[UnstructuredTextDataSet] = _get_edp_field("unstructuredTextDatasets")
    structuredDatasets: List[StructuredDataSet] = _get_edp_field("structuredDatasets")
    imageDatasets: List[ImageDataSet] = _get_edp_field("imageDatasets")
    compression: Compression | None = _get_edp_field("compression")
    schema_version: SchemaVersion = _get_edp_field("schema_version")
    volume: int = _get_edp_field("volume")
    videoDatasets: List[VideoDataSet] = _get_edp_field("videoDatasets")
    temporalCover: TemporalCover | None = _get_edp_field("temporalCover")
    periodicity: str | None = _get_edp_field("periodicity")
    documentDatasets: List[DocumentDataSet] = _get_edp_field("documentDatasets")


class AugmentedColumn(BaseModel):
    name: str = Field(description="Name of the augmented column")
    file: Optional[PurePosixPath] = Field(
        default=None,
        description="Path of the augmented file this column was added. If augmentedFile is None, EDPS will assume that the augmented column contained in all files.",
    )
    augmentation: Augmentation = Field(description="Augmentation information")


class Config(BaseModel):
    userProvidedEdpData: UserProvidedEdpData = Field(description="User provided EDP meta data")
    augmentedColumns: List[AugmentedColumn] = Field(default_factory=list, description="List of augmented columns")


def recursively_escape_strings(data: Any) -> Any:
    if data is None or isinstance(data, (bool, datetime)):
        return data
    elif isinstance(data, str):
        return html.escape(data)
    elif isinstance(data, list):
        return [recursively_escape_strings(item) for item in data]
    elif isinstance(data, dict):
        return {k: recursively_escape_strings(v) for k, v in data.items()}
    elif isinstance(data, BaseModel):
        as_dict = data.model_dump()
        escaped_dict = recursively_escape_strings(as_dict)
        return data.__class__(**escaped_dict)
    else:
        raise NotImplementedError(f"Type {type(data)} not supported")
