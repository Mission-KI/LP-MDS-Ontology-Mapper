from asyncio import get_running_loop
from csv import Sniffer
from logging import getLogger

from pandas import read_csv

from edp.analyzers.pandas import Pandas
from edp.file import File


async def csv(file: File):
    logger = getLogger("CSV Importer")
    logger.info("Importing %s as CSV", file)

    sniffer = Sniffer()
    with open(file.path.absolute(), "rt") as open_file:
        csv_snippet = open_file.read(5000)

    dialect = sniffer.sniff(csv_snippet)
    has_header = sniffer.has_header(csv_snippet)

    def runner():
        return read_csv(file.path.absolute(), sep=dialect.delimiter, header="infer" if has_header else None)

    loop = get_running_loop()
    data_frame = await loop.run_in_executor(None, runner)

    # Set column headers (col000, col001, ...) if the csv doesn't contain headers.
    # Otherwise the headers are just numbers, not strings.
    if not has_header:
        col_count = len(data_frame.columns)
        data_frame.columns = [f"col{i:03d}" for i in range(col_count)]

    return Pandas(data_frame, file)
