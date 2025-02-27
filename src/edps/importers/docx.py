from docx import Document
from extended_dataset_profile.models.v0.edp import DocumentDataSet

from edps.analyzers.docx import DocxAnalyzer
from edps.file import File
from edps.task import TaskContext


async def docx_importer(ctx: TaskContext, file: File) -> DocumentDataSet:
    ctx.logger.info("Analyzing DOCX '%s'", file)

    doc = Document(str(file.path))
    return await DocxAnalyzer(doc, file).analyze(ctx)
