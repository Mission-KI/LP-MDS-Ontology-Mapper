from typing import AsyncIterator, Callable

from edps.analyzers import Analyzer
from edps.file import File
from edps.importers.images import raster_image_importer
from edps.importers.pandas import csv_importer, xls_importer, xlsx_importer
from edps.importers.python import pickle_importer
from edps.task import TaskContext

Importer = Callable[[TaskContext, File], AsyncIterator[Analyzer]]

# Dictionary mapping a file extension to Importer
IMPORTERS: dict[str, Importer] = {
    "csv": csv_importer,
    "xls": xls_importer,
    "xlsx": xlsx_importer,
    "pickle": pickle_importer,
    "png": raster_image_importer,
    "jpg": raster_image_importer,
    "jpeg": raster_image_importer,
    "gif": raster_image_importer,
    "bmp": raster_image_importer,
    "tiff": raster_image_importer,
    "tif": raster_image_importer,
    "webp": raster_image_importer,
}
