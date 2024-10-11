from asyncio import get_running_loop
from logging import getLogger

from pandas import read_csv

from edp.analyzers.pandas import Pandas
from edp.file import File
from csv import Sniffer

async def csv(file: File):
    logger = getLogger("CSV Importer")
    logger.info("Importing %s as CSV", file)

    sniffer = Sniffer()
    with open(file.path.absolute(), "rt") as open_file:
        csv_snippet = open_file.read(5000)

    dialect = sniffer.sniff(csv_snippet)
    def runner():
        return read_csv(file.path.absolute(), sep=dialect.delimiter)

    loop = get_running_loop()
    data_frame = await loop.run_in_executor(None, runner)
    return Pandas(data_frame, file)
