from asyncio import get_running_loop
from pickle import load

from pandas import DataFrame

from edp.analyzers import Pandas as PandasAnalyzer
from edp.file import File
from edp.task import TaskContext


async def pickle(ctx: TaskContext, file: File):
    ctx.logger.info("Reading object from %s", file)
    loop = get_running_loop()
    with open(file.path, "rb") as opened_file:
        data_object = await loop.run_in_executor(None, load, opened_file)
    if isinstance(data_object, DataFrame):
        ctx.logger.info("Finished reading object, object recognized as pandas data frame")
        return PandasAnalyzer(data_object, file)

    raise NotImplementedError(f'Type "{type(data_object)}" not yet supported')
