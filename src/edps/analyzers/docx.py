import warnings
from pathlib import Path, PurePosixPath
from typing import AsyncIterator
from uuid import uuid4
from zipfile import ZipFile

from docx.document import Document
from extended_dataset_profile.models.v0.edp import (
    DataSet,
    DocumentDataSet,
    ImageDataSet,
    ModificationState,
)
from pandas import DataFrame

from edps.analyzers.common import split_keywords
from edps.importers.images import raster_image_importer
from edps.importers.structured import pandas_importer
from edps.importers.unstructured_text import unstructured_text_importer_from_string
from edps.task import TaskContext


class DocxAnalyzer:
    def __init__(self, doc: Document, path: Path):
        self.doc = doc
        self.path = path

    async def analyze(self, ctx: TaskContext) -> DocumentDataSet:
        props = self.doc.core_properties
        keywords = split_keywords(props.keywords)
        modified = props.revision > 1

        extracted_text = self._extract_text(ctx)
        await ctx.exec("text", unstructured_text_importer_from_string, extracted_text)

        num_images = 0
        async for media_file in self._extract_media_files(ctx):
            try:
                child_ds = await ctx.exec_with_result(media_file.name, self._analyze_media_file, media_file)
                if isinstance(child_ds, ImageDataSet):
                    num_images += 1
            except Exception as exception:
                message = f"Image importer can't process DOCX media file '{media_file}'"
                ctx.logger.warning(message, exc_info=exception)
                warnings.warn(message)

        num_tables = 0
        async for dataframe in self._extract_tables(ctx):
            num_tables += 1
            await ctx.exec(f"table_{num_images:03}", pandas_importer, dataframe)

        return DocumentDataSet(
            uuid=uuid4(),  # TODO uuid, parentUuid & name are set by the TaskContext and don't need explicit initialization!
            parentUuid=None,
            name=PurePosixPath(""),
            fileSize=0,  # TODO remove fileSize from DocumentDataSet because every DataSet has this as an optional property!
            title=props.title,
            subject=props.subject,
            author=props.author,
            toolchain=None,
            creationDate=props.created,
            modificationDate=props.modified,
            keywords=keywords,
            docType="DOCX",
            numPages=None,  # unknown, this would need rendering the pages
            numImages=num_images,
            modified=ModificationState.modified if modified else ModificationState.unmodified,
            encrypted=False,
        )

    def _extract_text(self, ctx: TaskContext) -> str:
        ctx.logger.info("Extracting text...")
        paragraphs = self.doc.paragraphs
        paragraph_texts = [p.text for p in paragraphs]
        return "\n".join(paragraph_texts)

    async def _extract_media_files(self, ctx: TaskContext) -> AsyncIterator[Path]:
        ctx.logger.info("Extracting media files...")
        # Open the .docx file as a zip archive
        with ZipFile(self.path, "r") as docx_zip:
            # Filter files in the 'word/media/' folder (which contains images)
            file_list = docx_zip.namelist()
            media_files = [file for file in file_list if file.startswith("word/media/")]
            if not media_files:
                return

            unzipped_media_dir = ctx.create_working_dir("media")
            ctx.logger.debug("Unzipping %d media files...", len(media_files))
            for media_file in media_files:
                unzipped_media_file_path = Path(unzipped_media_dir) / PurePosixPath(media_file).name
                with unzipped_media_file_path.open("wb") as file_io:
                    # Unzip the image to a temp file
                    file_io.write(docx_zip.read(media_file))
                    ctx.logger.debug("Extracted '%s' from DOCX to '%s'", media_file, unzipped_media_file_path)
                yield unzipped_media_file_path

    async def _extract_tables(self, ctx: TaskContext) -> AsyncIterator[DataFrame]:
        ctx.logger.info("Extracting tables...")
        for table in self.doc.tables:
            cells: list[list[str]] = []
            for row in table.rows:
                row_cells = [c.text for c in row.cells]
                cells.append(row_cells)
            # Assume first row is header
            yield DataFrame(cells[1:], columns=cells[0])

    async def _analyze_media_file(self, ctx: TaskContext, media_file: Path) -> DataSet:
        ctx.logger.info("Analyzing DOCX embedded media file '%s'...", ctx.relative_path(media_file))
        # TODO We can't use the import dictionary yet because we would get a circular dependency. To resolve this we'll move this into the TaskContext.
        return await raster_image_importer(ctx, media_file)
