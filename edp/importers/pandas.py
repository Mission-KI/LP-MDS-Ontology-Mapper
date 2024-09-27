from pandas import read_csv

from edp.analyzers.pandas import Pandas
from edp.file import File


async def csv(file: File):
    return Pandas(read_csv(file.path.absolute()))
