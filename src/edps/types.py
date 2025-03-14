import html
from datetime import datetime
from typing import Any, List, Optional, Set, Union, get_args

from extended_dataset_profile import SchemaVersion
from extended_dataset_profile.models.v0.edp import (
    ArchiveDataSet,
    AssetProcessingStatus,
    AssetReference,
    Augmentation,
    DataSetFrequency,
    DataSetImmutability,
    DataSetTransfer,
    DatasetTreeNode,
    DataSetType,
    DataSetVolume,
    DocumentDataSet,
    ExtendedDatasetProfile,
    ImageDataSet,
    SemiStructuredDataSet,
    StructuredDataSet,
    TemporalCover,
    UnstructuredTextDataSet,
    VideoDataSet,
)
from pydantic import AnyUrl, BaseModel, Field, model_validator


def _get_edp_field(name: str) -> Any:
    """Function to avoid mypy from complaining about putting FieldInfo's into fields."""
    return ExtendedDatasetProfile.model_fields[name]


class UserProvidedEdpData(BaseModel):
    """
    The data provided by the caller of the EDP Service.

    This is a subset of the extended dataset profile.
    """

    name: str = _get_edp_field("name")
    assetRefs: List[AssetReference] = _get_edp_field("assetRefs")
    dataCategory: str | None = _get_edp_field("dataCategory")
    assetProcessingStatus: AssetProcessingStatus | None = _get_edp_field("assetProcessingStatus")
    description: str | None = _get_edp_field("description")
    tags: List[str] = _get_edp_field("tags")
    dataSubCategory: str | None = _get_edp_field("dataSubCategory")
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
    archiveDatasets: List[ArchiveDataSet] = _get_edp_field("archiveDatasets")
    structuredDatasets: List[StructuredDataSet] = _get_edp_field("structuredDatasets")
    semiStructuredDatasets: List[SemiStructuredDataSet] = _get_edp_field("semiStructuredDatasets")
    unstructuredTextDatasets: List[UnstructuredTextDataSet] = _get_edp_field("unstructuredTextDatasets")
    imageDatasets: List[ImageDataSet] = _get_edp_field("imageDatasets")
    schema_version: SchemaVersion = _get_edp_field("schema_version")
    volume: int = _get_edp_field("volume")
    videoDatasets: List[VideoDataSet] = _get_edp_field("videoDatasets")
    temporalCover: TemporalCover | None = _get_edp_field("temporalCover")
    periodicity: str | None = _get_edp_field("periodicity")
    documentDatasets: List[DocumentDataSet] = _get_edp_field("documentDatasets")
    datasetTree: List[DatasetTreeNode] = _get_edp_field("datasetTree")


class AugmentedColumn(BaseModel):
    name: str = Field(description="Name of the augmented column")
    datasetName: Optional[str] = Field(
        default=None,
        description="Name of the dataset this column was added to. If datasetName is None, EDPS will assume that the augmented column contained in all structured datasets.",
    )
    augmentation: Augmentation = Field(description="Augmentation information")


class DistributionConfig(BaseModel):
    """Configuration parameters specific to the distribution analysis."""

    minimum_number_unique: int = Field(
        default=16, description="Minimum number of unique values to run distribution analysis"
    )


class SeasonalityConfig(BaseModel):
    """Configuration for the seasonality analysis step."""

    max_samples: int = Field(default=int(1e6), description="Maximum number of samples to use for seasonality analysis.")
    max_periods: int = Field(
        default=10,
        description="Maximum number of periods to use in seasonality analysis. If this is too hight, the seasonality graphs become none readable.",
    )


class StructuredConfig(BaseModel):
    """Configurations for the structured data analysis"""

    distribution: DistributionConfig = Field(
        default_factory=DistributionConfig,
        description="Configuration parameters specific to the distribution analysis.",
    )
    seasonality: SeasonalityConfig = Field(
        default_factory=SeasonalityConfig, description="Configurations specific to the seasonality analysis."
    )


class Config(BaseModel):
    userProvidedEdpData: UserProvidedEdpData = Field(description="User provided EDP meta data")
    augmentedColumns: List[AugmentedColumn] = Field(default_factory=list, description="List of augmented columns")
    structured_config: StructuredConfig = Field(
        default_factory=StructuredConfig, description="Configurations for the structured data analysis"
    )


def recursively_escape_strings(data: Any) -> Any:
    if data is None or isinstance(data, (bool, datetime, AnyUrl)):
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


DataSet = Union[
    ArchiveDataSet,
    StructuredDataSet,
    SemiStructuredDataSet,
    UnstructuredTextDataSet,
    ImageDataSet,
    VideoDataSet,
    DocumentDataSet,
]


def is_dataset(value: Any) -> bool:
    dataset_types = get_args(DataSet)
    return any(isinstance(value, dataset_type) for dataset_type in dataset_types)
