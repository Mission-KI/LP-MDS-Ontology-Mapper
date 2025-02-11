from pathlib import PurePosixPath
from typing import AsyncIterator, Iterator, Optional
from uuid import uuid4

from extended_dataset_profile.models.v0.edp import DataSet, DocumentDataSet, ImageDataSet, ModificationState
from PIL.Image import Image
from pypdf import DocumentInformation, PdfReader
from pypdf.generic import ArrayObject, PdfObject

from edps.analyzers.base import Analyzer
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
        keywords = _calc_keywords(metadata.keywords if metadata else None)
        modified = _calc_modified(self.pdf_reader._ID, metadata)

        self._extract_text(ctx)
        # TODO yield TextDataSet from TextImporter/Analyzer
        # extracted_text = self._extract_text(ctx)
        # async for dataset in self._analyze_text(ctx, extracted_text):
        #     dataset.parentUuid = doc_uuid
        #     yield dataset

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
        page_text = [p.extract_text() for p in pages]
        return "\n".join(page_text)

    def _extract_images(self, ctx: TaskContext) -> Iterator[Image]:
        ctx.logger.info("Extracting images...")
        pages = self.pdf_reader.pages
        for page in pages:
            for image_file in page.images:
                image = image_file.image
                if image:
                    yield image

    # async def _analyze_text(self, ctx: TaskContext, text: str) -> AsyncIterator[DataSet]:
    #     yield ...

    async def _analyze_image(self, ctx: TaskContext, image: Image, count: int) -> AsyncIterator[ImageDataSet]:
        name = self.file.relative / f"image{count:03}"
        ctx.logger.info("Invoking ImageAnalyzer for image #%d", count)
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


def _calc_keywords(keywords_raw: Optional[str]) -> list[str]:
    keywords_raw = keywords_raw.strip() if keywords_raw else None
    # Ignore one-character keyword strings, like "|"
    if not keywords_raw or len(keywords_raw) <= 1:
        return []

    # Try splitting on ";", then on ",", then on any whitespace character.
    # Splitting must produce at least two parts, otherwise fallback to full keywords_raw.
    return (
        _try_splitting(keywords_raw, ";")
        or _try_splitting(keywords_raw, ",")
        or _try_splitting(keywords_raw, None)
        or [keywords_raw]
    )


def _try_splitting(text: str, sep: str | None) -> list[str] | None:
    # Split on separator, strip each part, remove empty parts.
    # Return only if splitting produced at least two parts.
    parts = text.split(sep)
    parts = [p.strip() for p in parts]
    parts = [p for p in parts if p]
    return parts if len(parts) > 1 else None
