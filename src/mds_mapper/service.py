from mimetypes import types_map
from pathlib import Path
from typing import Iterator, List, Optional

from extended_dataset_profile import AssetReference
from extended_dataset_profile import __version__ as EDP_VERSION
from iso639 import Language
from pydantic import ValidationError

from .types import (
    DataCategory,
    DataSubCategory,
    DateRange,
    ExtendedDatasetProfile,
    MobilityDataSpaceOntology,
    TemporalCoverage,
)


async def convert_edp_to_mds(edp_path: Path, output_mds_path: Path) -> None:
    edp_json = edp_path.read_bytes()
    try:
        edp = ExtendedDatasetProfile.model_validate_json(edp_json)
    except ValidationError as error:
        raise RuntimeError(
            f"Unable to import the supplied file as ExtendedDatasetProfile version {EDP_VERSION}."
        ) from error

    if len(edp.assetRefs) == 0:
        raise RuntimeError("EDP needs to have at least one assetRefs to be convertible!")
    asset_ref: AssetReference = edp.assetRefs[0]

    mds = MobilityDataSpaceOntology(
        name=edp.name,
        version=asset_ref.assetVersion,
        asset_id=asset_ref.assetId,
        description=edp.description,
        keywords=edp.tags,
        languages=_extract_languages(edp),
        content_type=_extract_content_type(edp),
        endpoint_documentation=None,
        publisher=asset_ref.publisher.url,
        organization=asset_ref.publisher.name,
        standard_licence=asset_ref.license.name if asset_ref.license.name else asset_ref.license.url,
        data_category=_extract_category(edp),
        data_subcategory=_extract_sub_category(edp),
        data_model=None,
        transport_mode=None,
        geo_reference_model=None,
        sovereign=asset_ref.publisher.name,
        data_update_frequency=edp.transferTypeFrequency.value if edp.transferTypeFrequency else None,
        geo_location=None,
        nuts_location=None,
        data_samples=None,
        reference_files=None,
        temporal_coverage=_extract_temporal_cover(edp),
        condition_for_use=_extract_conditions_for_use(edp),
    )
    output_mds_path.write_text(mds.model_dump_json(by_alias=True), encoding="utf-16")


def _extract_languages(edp: ExtendedDatasetProfile) -> Optional[str]:
    languages = set(_iterate_languages(edp))
    count = len(languages)
    if count == 0:
        return None
    elif count == 1:
        return languages.pop()
    else:
        return "Multilingual"


def _iterate_languages(edp: ExtendedDatasetProfile) -> Iterator[str]:
    for dataset in edp.unstructuredTextDatasets:
        for language in dataset.languages:
            yield Language.from_part3(language).name


def _extract_content_type(edp: ExtendedDatasetProfile) -> Optional[str]:
    file_extensions = list(_iterate_file_extensions(edp))
    mime_types = [types_map.get(f".{file_extension}") for file_extension in file_extensions]
    non_null_mime_types = set([mime_type for mime_type in mime_types if mime_type is not None])
    if len(mime_types) == 0:
        return None
    return ", ".join(non_null_mime_types)


def _iterate_file_extensions(edp: ExtendedDatasetProfile) -> Iterator[str]:
    for node in edp.datasetTree:
        if node.fileProperties:
            yield node.fileProperties.fileType


def _extract_category(edp: ExtendedDatasetProfile) -> DataCategory:
    if edp.dataCategory is None:
        return DataCategory.Various

    for category in DataCategory:
        if edp.dataCategory.lower() == category.value.lower():
            return category

    return DataCategory.Various


def _extract_sub_category(edp: ExtendedDatasetProfile) -> Optional[DataSubCategory]:
    if edp.dataSubCategory is None:
        return None

    for category in DataSubCategory:
        if edp.dataSubCategory.lower() == category.value.lower():
            return category

    return None


def _extract_temporal_cover(edp: ExtendedDatasetProfile) -> TemporalCoverage:
    if edp.temporalCover is None:
        return None
    return DateRange(start=edp.temporalCover.earliest.date(), end=edp.temporalCover.latest.date())


def _extract_conditions_for_use(edp: ExtendedDatasetProfile) -> Optional[str]:
    conditions: List[str] = []
    if edp.nda:
        conditions.append(f"NDA: {edp.nda}")
    if edp.allowedForAiTraining:
        conditions.append(f"Allowed for AI Training: {'Yes' if edp.allowedForAiTraining else 'No'}")
    if edp.dpa:
        conditions.append(f"DPA: {edp.dpa}")
    if len(conditions) == 0:
        return None
    return ", ".join(conditions)
