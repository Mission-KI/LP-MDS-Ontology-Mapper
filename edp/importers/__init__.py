from typing import Awaitable as _Awaitable
from typing import Callable as _Callable

from edp.analyzers import Analyzer as _Analyzer
from edp.file import File as _File
from edp.importers.pandas import csv_importer, xls_importer, xlsx_importer
from edp.importers.python import pickle as pickle_importer

Importer = _Callable[[_File], _Awaitable[_Analyzer]]

# Dictionary mapping a file extension to Importer
IMPORTERS: dict[str, Importer] = {
    "csv": csv_importer,
    "xls": xls_importer,
    "xlsx": xlsx_importer,
    "pickle": pickle_importer,
}
