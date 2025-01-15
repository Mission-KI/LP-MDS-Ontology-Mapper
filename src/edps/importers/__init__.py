from typing import Awaitable as _Awaitable
from typing import Callable as _Callable

from edps.analyzers import Analyzer as _Analyzer
from edps.file import File as _File
from edps.importers.pandas import csv_importer, xls_importer, xlsx_importer
from edps.importers.python import pickle as pickle_importer
from edps.task import TaskContext

Importer = _Callable[[TaskContext, _File], _Awaitable[_Analyzer]]

# Dictionary mapping a file extension to Importer
IMPORTERS: dict[str, Importer] = {
    "csv": csv_importer,
    "xls": xls_importer,
    "xlsx": xlsx_importer,
    "pickle": pickle_importer,
}
