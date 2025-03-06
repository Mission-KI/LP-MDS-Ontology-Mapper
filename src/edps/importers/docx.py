from pathlib import Path

from docx import Document
from extended_dataset_profile.models.v0.edp import DocumentDataSet

from edps.analyzers.docx import DocxAnalyzer
from edps.task import TaskContext


async def docx_importer(ctx: TaskContext, path: Path) -> DocumentDataSet:
    ctx.logger.info("Analyzing DOCX '%s'", ctx.relative_path(path))

    doc = Document(str(path))
    return await DocxAnalyzer(doc, path).analyze(ctx)
