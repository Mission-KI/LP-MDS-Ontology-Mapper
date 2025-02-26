import warnings
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import AsyncIterator
from uuid import uuid4
from zipfile import ZipFile

from docx.document import Document
from extended_dataset_profile.models.v0.edp import (
    DataSet,
    DocumentDataSet,
    ImageDataSet,
    ModificationState,
    StructuredDataSet,
)
from pandas import DataFrame

from edps.analyzers.base import Analyzer
from edps.analyzers.common import split_keywords
from edps.analyzers.pandas import PandasAnalyzer
from edps.analyzers.unstructured_text import Analyzer as UnstructuredTextAnalyzer
from edps.file import File
from edps.importers.images import raster_image_importer
from edps.task import TaskContext


class DocxAnalyzer(Analyzer):
    def __init__(self, doc: Document, file: File):
        self.doc = doc
        self.file = file

    async def analyze(self, ctx: TaskContext) -> AsyncIterator[DataSet]:
        ctx.logger.info("Analyzing DOCX document '%s'...", self.file)

        doc_uuid = uuid4()
        props = self.doc.core_properties
        keywords = split_keywords(props.keywords)
        modified = props.revision > 1

        extracted_text = self._extract_text(ctx)
        async for dataset in self._analyze_text(ctx, extracted_text):
            dataset.parentUuid = doc_uuid
            yield dataset

        num_images = 0
        async for media_file in self._extract_media_files(ctx):
            async for child_ds in self._analyze_media_file(ctx, media_file):
                child_ds.parentUuid = doc_uuid
                if isinstance(child_ds, ImageDataSet):
                    num_images += 1
                yield child_ds

        num_tables = 0
        async for dataframe in self._extract_tables(ctx):
            num_tables += 1
            async for structured_ds in self._analyze_table(ctx, dataframe, num_tables):
                structured_ds.parentUuid = doc_uuid
                yield structured_ds

        yield DocumentDataSet(
            uuid=doc_uuid,
            parentUuid=None,
            name=PurePosixPath(self.file.relative),
            fileSize=self.file.size,
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

    async def _extract_media_files(self, ctx: TaskContext) -> AsyncIterator[File]:
        ctx.logger.info("Extracting media files...")
        # Open the .docx file as a zip archive
        with ZipFile(self.file.path, "r") as docx_zip:
            # Filter files in the 'word/media/' folder (which contains images)
            file_list = docx_zip.namelist()
            media_files = [file for file in file_list if file.startswith("word/media/")]
            if not media_files:
                return

            with TemporaryDirectory() as tmp_dir:
                ctx.logger.debug("Unzipping %d media files...", len(media_files))
                for media_file in media_files:
                    media_file_name = PurePosixPath(media_file).name
                    media_file_path = Path(tmp_dir) / media_file_name
                    with media_file_path.open("wb") as file_io:
                        # Unzip the image to a temp file
                        file_io.write(docx_zip.read(media_file))
                        ctx.logger.debug("Extracted '%s' from DOCX to '%s'", media_file, media_file_path)
                    yield File(media_file_path.parent, media_file_path)

    async def _extract_tables(self, ctx: TaskContext) -> AsyncIterator[DataFrame]:
        ctx.logger.info("Extracting tables...")
        for table in self.doc.tables:
            cells: list[list[str]] = []
            for row in table.rows:
                row_cells = [c.text for c in row.cells]
                cells.append(row_cells)
            # Assume first row is header
            yield DataFrame(cells[1:], columns=cells[0])

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

    async def _analyze_media_file(self, ctx: TaskContext, media_file: File) -> AsyncIterator[DataSet]:
        ctx.logger.info("Importing embedded media file '%s'...", media_file)
        # TODO We can't use the import dictionary yet because we would get a circular dependency. To resolve this we'll move this into the TaskContext.
        try:
            async for analyzer in raster_image_importer(ctx, media_file):
                async for ds in ctx.exec(analyzer.analyze):
                    yield ds
        except Exception as exception:
            message = f"Image importer can't process DOCX media file '{media_file}'"
            ctx.logger.warning(message, exc_info=exception)
            warnings.warn(message)

    async def _analyze_table(self, ctx: TaskContext, table: DataFrame, count: int) -> AsyncIterator[StructuredDataSet]:
        ctx.logger.info("Invoking structured data analyzer for table %d", count)
        # TODO pass virtual file name!
        analyzer = PandasAnalyzer(table, self.file)
        async for ds in ctx.exec(analyzer.analyze):
            yield ds
