from extended_dataset_profile.models.v0.edp import UnstructuredTextDataSet

from edps.analyzers.unstructured_text import Analyzer as UnstructuredTextAnalyzer
from edps.file import File
from edps.task import TaskContext


async def unstructured_text_importer(ctx: TaskContext, file: File) -> UnstructuredTextDataSet:
    ctx.logger.info("Analyzing unstructured text '%s'", ctx.dataset_name)
    return await UnstructuredTextAnalyzer(file).analyze(ctx)
