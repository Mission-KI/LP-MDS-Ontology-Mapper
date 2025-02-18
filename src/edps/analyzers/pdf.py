import warnings
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import AsyncIterator, Iterator, Optional
from uuid import uuid4

from extended_dataset_profile.models.v0.edp import DataSet, DocumentDataSet, ImageDataSet, ModificationState
from PIL.Image import Image
from pypdf import DocumentInformation, PageObject, PdfReader
from pypdf.constants import PageAttributes
from pypdf.generic import ArrayObject, PdfObject

from edps.analyzers.base import Analyzer
from edps.analyzers.common import split_keywords
from edps.analyzers.unstructured_text import Analyzer as UnstructuredTextAnalyzer
from edps.file import File
from edps.importers.images import build_raster_image_analyzer
from edps.task import TaskContext


class PdfAnalyzer(Analyzer):
    def __init__(self, pdf_reader: PdfReader, file: File):
        self.pdf_reader = pdf_reader
        self.file = file

    async def analyze(self, ctx: TaskContext) -> AsyncIterator[DataSet]:
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
        async for dataset in self._analyze_text(ctx, extracted_text):
            dataset.parentUuid = doc_uuid
            yield dataset

        num_images = 0
        for image in self._extract_images(ctx):
            num_images += 1
            async for dataset in self._analyze_image(ctx, image, num_images):
                dataset.parentUuid = doc_uuid
                yield dataset

        yield DocumentDataSet(
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

    async def _analyze_text(self, ctx: TaskContext, text: str) -> AsyncIterator[DataSet]:
        ctx.logger.info("Invoking unstructured text analyzer")
        # TODO(mka): Rework this after Datasets are added to TaskContext
        with TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / "content.txt"
            # UTF-8 is needed for umlauts etc.
            with tmp_file.open("wt", encoding="utf-8") as file_io:
                file_io.write(text)
            file = File(tmp_file.parent, tmp_file)
            text_analyzer = UnstructuredTextAnalyzer(file)
            async for dataset in ctx.exec(text_analyzer.analyze):
                yield dataset

    async def _analyze_image(self, ctx: TaskContext, image: Image, count: int) -> AsyncIterator[ImageDataSet]:
        name = self.file.relative / f"image{count:03}"
        ctx.logger.info("Invoking image analyzer for image #%d", count)
        img_analyzer = build_raster_image_analyzer(image, PurePosixPath(name))
        async for dataset in ctx.exec(img_analyzer.analyze):
            yield dataset


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
