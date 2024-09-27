from pandas import read_csv

from edp.file import File
from edp.analyzers.pandas import Pandas


async def csv(file: File):
    return Pandas(read_csv(file.path.absolute()))
