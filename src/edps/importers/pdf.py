from pypdf import PdfReader

from edps.analyzers.pdf import PdfAnalyzer
from edps.file import File
from edps.task import TaskContext


async def pdf_importer(ctx: TaskContext, file: File):
    ctx.logger.info("Importing '%s' as PDF", file)

    with PdfReader(file.path) as pdf_reader:
        yield PdfAnalyzer(pdf_reader, file)
