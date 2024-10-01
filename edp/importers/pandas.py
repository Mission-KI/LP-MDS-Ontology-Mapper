from asyncio import get_running_loop
from logging import getLogger

from pandas import read_csv

from edp.analyzers.pandas import Pandas
from edp.file import File


async def csv(file: File):
    logger = getLogger("CSV Importer")
    logger.info("Importing %s as CSV", file)
    loop = get_running_loop()
    data_frame = await loop.run_in_executor(None, read_csv, file.path.absolute())
    return Pandas(data_frame)
