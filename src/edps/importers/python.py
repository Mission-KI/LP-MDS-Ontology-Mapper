from asyncio import get_running_loop
from pickle import load

from extended_dataset_profile.models.v0.edp import StructuredDataSet
from pandas import DataFrame

from edps.analyzers import PandasAnalyzer
from edps.file import File
from edps.task import TaskContext


async def pickle_importer(ctx: TaskContext, file: File) -> StructuredDataSet:
    ctx.logger.info("Reading object from %s", file)
    loop = get_running_loop()
    with open(file.path, "rb") as opened_file:
        data_object = await loop.run_in_executor(None, load, opened_file)
    if not isinstance(data_object, DataFrame):
        raise NotImplementedError(f'Type "{type(data_object)}" not yet supported')
    ctx.logger.info("Finished reading object, object recognized as pandas data frame")
    return await PandasAnalyzer(data_object, file).analyze(ctx)
