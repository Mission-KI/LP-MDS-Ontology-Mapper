from typing import Awaitable as _Awaitable
from typing import Callable as _Callable

from edp.analyzers import Analyzer as _Analyzer
from edp.file import File as _File

Importer = _Callable[[_File], _Awaitable[_Analyzer]]

from edp.importers.pandas import csv as csv_importer
