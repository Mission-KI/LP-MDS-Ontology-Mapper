from extended_dataset_profile.models.v0.edp import DocumentDataSet
from pypdf import PdfReader

from edps.analyzers.pdf import PdfAnalyzer
from edps.file import File
from edps.task import TaskContext


async def pdf_importer(ctx: TaskContext, file: File) -> DocumentDataSet:
    ctx.logger.info("Analyzing PDF '%s'", file)

    with PdfReader(file.path) as pdf_reader:
        return await PdfAnalyzer(pdf_reader, file).analyze(ctx)
