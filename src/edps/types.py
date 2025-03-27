import html
from datetime import datetime, timedelta
from typing import Any, List, Optional, Set, Union, get_args

from extended_dataset_profile.models.v0.edp import (
    ArchiveDataSet,
    AssetGrowthRate,
    AssetImmutability,
    AssetProcessingStatus,
    AssetReference,
    AssetTransferType,
    AssetUpdatePeriod,
    Augmentation,
    DatasetTreeNode,
    DataSetType,
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
    All fields of the extended dataset profile which must be supplied by the caller.
    """

    name: str = _get_edp_field("name")
    assetRefs: List[AssetReference] = _get_edp_field("assetRefs")
    dataCategory: str | None = _get_edp_field("dataCategory")
    assetProcessingStatus: AssetProcessingStatus | None = _get_edp_field("assetProcessingStatus")
    description: str | None = _get_edp_field("description")
    tags: List[str] = _get_edp_field("tags")
    dataSubCategory: str | None = _get_edp_field("dataSubCategory")
    assetTypeInfo: str | None = _get_edp_field("assetTypeInfo")
    transferTypeFlag: AssetTransferType | None = _get_edp_field("transferTypeFlag")
    transferTypeFrequency: AssetUpdatePeriod | None = _get_edp_field("transferTypeFrequency")
    growthFlag: AssetGrowthRate | None = _get_edp_field("growthFlag")
    immutabilityFlag: AssetImmutability | None = _get_edp_field("immutabilityFlag")
    nda: str | None = _get_edp_field("nda")
    dpa: str | None = _get_edp_field("dpa")
    dataLog: str | None = _get_edp_field("dataLog")
    freely_available: bool = _get_edp_field("freely_available")

    @model_validator(mode="before")
    def escape_all_string_fields(cls, data: Any) -> Any:
        return recursively_escape_strings(data)


class ComputedEdpData(BaseModel):
    """All fields of the extended dataset profile that get calculated by this service."""

    generatedBy: str = _get_edp_field("generatedBy")
    dataTypes: Set[DataSetType] = _get_edp_field("dataTypes")
    assetSha256Hash: str = _get_edp_field("assetSha256Hash")
    archiveDatasets: List[ArchiveDataSet] = _get_edp_field("archiveDatasets")
    structuredDatasets: List[StructuredDataSet] = _get_edp_field("structuredDatasets")
    semiStructuredDatasets: List[SemiStructuredDataSet] = _get_edp_field("semiStructuredDatasets")
    unstructuredTextDatasets: List[UnstructuredTextDataSet] = _get_edp_field("unstructuredTextDatasets")
    imageDatasets: List[ImageDataSet] = _get_edp_field("imageDatasets")
    schemaVersion: str = _get_edp_field("schemaVersion")
    volume: int = _get_edp_field("volume")
    videoDatasets: List[VideoDataSet] = _get_edp_field("videoDatasets")
    temporalCover: TemporalCover | None = _get_edp_field("temporalCover")
    periodicity: str | None = _get_edp_field("periodicity")
    documentDatasets: List[DocumentDataSet] = _get_edp_field("documentDatasets")
    datasetTree: List[DatasetTreeNode] = _get_edp_field("datasetTree")


class AugmentedColumn(BaseModel):
    """
    Stores information about a column having been modified before running the service.
    This information will be attached to each column on service execution.
    """

    name: str = Field(description="Name of the augmented column")
    datasetName: Optional[str] = Field(
        default=None,
        description="Name of the dataset this column was added to. If datasetName is None, EDPS will assume that the augmented column contained in all structured datasets.",
    )
    augmentation: Augmentation = Field(description="Augmentation information")


class DistributionConfig(BaseModel):
    """Configuration parameters specific to the distribution analysis."""

    minimum_number_numeric_values: int = Field(
        default=16, description="Minimum number of interpretable values to run numeric distribution analysis"
    )
    minimum_number_unique_string: int = Field(
        default=4, description="Minimum number of unique values to run string distribution analysis"
    )
    timeout: timedelta = Field(default=timedelta(minutes=1), description="Timeout to use for the distribution fitting.")
    max_samples: int = Field(
        default=int(1e6), description="Maximum number of values to use for determining the distribution of values."
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


class UnstructuredTextConfig(BaseModel):
    """
    Configuration for the unstructured text analysis.
    """

    minimum_sentence_length: int = Field(
        default=14,
        description="Minimum number of words in sentence for language detection. Shorter sentences will be skipped.",
    )
    language_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence to count a language as detected."
    )


class Config(BaseModel):
    """
    Extended dataset profile service configuration

    This configuration contains all customizable variables for the analysis of assets.
    All analyzer configurations are collected here.
    """

    augmentedColumns: List[AugmentedColumn] = Field(default_factory=list, description="List of augmented columns")
    structured_config: StructuredConfig = Field(
        default_factory=StructuredConfig, description="Configurations for the structured data analysis"
    )
    unstructured_text_config: UnstructuredTextConfig = Field(
        default_factory=lambda: UnstructuredTextConfig(),
        description="Configuration for the unstructured text analysis.",
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
