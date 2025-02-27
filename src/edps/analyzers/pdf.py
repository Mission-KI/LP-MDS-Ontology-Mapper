import warnings
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Iterator, Optional
from uuid import uuid4

from extended_dataset_profile.models.v0.edp import (
    DocumentDataSet,
    ModificationState,
    UnstructuredTextDataSet,
)
from PIL.Image import Image
from pypdf import DocumentInformation, PageObject, PdfReader
from pypdf.constants import PageAttributes
from pypdf.generic import ArrayObject, PdfObject

from edps.analyzers.common import split_keywords
from edps.file import File
from edps.importers.images import raster_image_importer_from_pilimage
from edps.importers.unstructured_text import unstructured_text_importer
from edps.task import TaskContext


class PdfAnalyzer:
    def __init__(self, pdf_reader: PdfReader, file: File):
        self.pdf_reader = pdf_reader
        self.file = file

    async def analyze(self, ctx: TaskContext) -> DocumentDataSet:
        ctx.logger.info("Analyzing PDF document '%s'...", self.file)

        doc_uuid = uuid4()
        metadata = self.pdf_reader.metadata
        # PDF header is something like that: %PDF-1.6
        file_version = self.pdf_reader.pdf_header.replace("%", "")
        pages = self.pdf_reader.pages
        toolchain = _calc_toolchain(metadata)
        keywords = split_keywords(metadata.keywords if metadata else None)
        modified = _calc_modified(self.pdf_reader._ID, metadata)

        extracted_text = self._extract_text(ctx)
        await ctx.exec("text", self._analyze_text, extracted_text)

        num_images = 0
        for image in self._extract_images(ctx):
            num_images += 1
            await ctx.exec(f"image_{num_images:03}", raster_image_importer_from_pilimage, image)

        # TODO: Change DocumentDataSet: give defaults for "uuid", "parent"; maybe make "name" optional and of type str
        return DocumentDataSet(
            uuid=doc_uuid,
            parentUuid=None,
            name=PurePosixPath(self.file.relative),
            fileSize=self.file.size,
            title=metadata.title if metadata is not None else None,
            subject=metadata.subject if metadata is not None else None,
            author=metadata.author if metadata is not None else None,
            toolchain=toolchain,
            creationDate=metadata.creation_date if metadata is not None else None,
            modificationDate=metadata.modification_date if metadata is not None else None,
            keywords=keywords,
            docType=file_version,
            numPages=len(pages),
            numImages=num_images,
            modified=modified,
            encrypted=self.pdf_reader.is_encrypted,
        )

    def _extract_text(self, ctx: TaskContext) -> str:
        ctx.logger.info("Extracting text...")
        pages = self.pdf_reader.pages
        page_text = [_extract_page_text(ctx, p) for p in pages]
        return "\n\n".join(page_text)

    def _extract_images(self, ctx: TaskContext) -> Iterator[Image]:
        ctx.logger.info("Extracting images...")
        pages = self.pdf_reader.pages
        for page in pages:
            for image_file in page.images:
                image = image_file.image
                if image:
                    yield image

    async def _analyze_text(self, ctx: TaskContext, text: str) -> UnstructuredTextDataSet:
        # TODO(mka): Rework this after Datasets are added to TaskContext
        with TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / ctx.qualified_path
            tmp_file.parent.mkdir(parents=True, exist_ok=True)
            # UTF-8 is needed for umlauts etc.
            with tmp_file.open("wt", encoding="utf-8") as file_io:
                file_io.write(text)
            file = File(Path(tmp_dir), tmp_file)
            return await unstructured_text_importer(ctx, file)


def _calc_modified(ids: Optional[ArrayObject], metadata: Optional[DocumentInformation]) -> ModificationState:
    modification_state = ModificationState.unknown
    # Dependending if the IDs are equal, set state to unmodified or modified
    if ids and len(ids) == 2:
        initial_id = ids[0]
        current_id = ids[1]
        if isinstance(initial_id, PdfObject) and isinstance(current_id, PdfObject):
            modified = initial_id.hash_bin() != current_id.hash_bin()
            modification_state = ModificationState.modified if modified else ModificationState.unmodified
    # Additionally set state to modified if creation_date and modification_date are different
    if metadata and metadata.creation_date != metadata.modification_date:
        modification_state = ModificationState.modified
    return modification_state


def _calc_toolchain(metadata: Optional[DocumentInformation]) -> Optional[str]:
    if not metadata:
        return None
    # Join fields "creator" and "producer" if they are non-empty and different
    parts = [p.strip() for p in [metadata.creator, metadata.producer] if p and p.strip()]
    # Remove duplicates
    parts = list(dict.fromkeys(parts))
    return "; ".join(parts) if parts else None


def _extract_page_text(ctx: TaskContext, page: PageObject) -> str:
    # PyPDF doesn't handle empty pages properly!
    if PageAttributes.CONTENTS not in page:
        return ""
    try:
        # Try layout extraction mode which aims to reproduce the text layout from the coordinates.
        # This is still experimental, so fall back to plain mode on error.
        return page.extract_text(extraction_mode="layout")
    except Exception as exception:
        message = "Error with PDF extraction in layout mode, falling back to plain mode..."
        ctx.logger.warning(message, exc_info=exception)
        warnings.warn(message)
        return page.extract_text(extraction_mode="plain")
