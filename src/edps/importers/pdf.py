from pathlib import Path

from extended_dataset_profile.models.v0.edp import DocumentDataSet
from pypdf import PdfReader

from edps.analyzers.pdf import PdfAnalyzer
from edps.task import TaskContext


async def pdf_importer(ctx: TaskContext, path: Path) -> DocumentDataSet:
    ctx.logger.info("Analyzing PDF '%s'", ctx.relative_path(path))

    with PdfReader(path) as pdf_reader:
        return await PdfAnalyzer(pdf_reader).analyze(ctx)
