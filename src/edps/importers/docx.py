from typing import AsyncIterator

from docx import Document

from edps.analyzers.docx import DocxAnalyzer
from edps.file import File
from edps.task import TaskContext


async def docx_importer(ctx: TaskContext, file: File) -> AsyncIterator[DocxAnalyzer]:
    ctx.logger.info("Importing '%s' as DOCX", file)

    doc = Document(str(file.path))
    yield DocxAnalyzer(doc, file)
