from typing import AsyncIterator

from edps.analyzers.base import Analyzer
from edps.analyzers.unstructured_text import Analyzer as UnstructuredTextAnalyzer
from edps.file import File
from edps.task import TaskContext


async def unstructured_text_importer(ctx: TaskContext, file: File) -> AsyncIterator[Analyzer]:
    yield UnstructuredTextAnalyzer(file)
