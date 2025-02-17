from typing import AsyncIterator, Callable

from edps.analyzers import Analyzer
from edps.file import File
from edps.importers.images import raster_image_importer
from edps.importers.pdf import pdf_importer
from edps.importers.python import pickle_importer
from edps.importers.structured import csv_importer, xls_importer, xlsx_importer
from edps.importers.unstructured_text import unstructured_text_importer
from edps.importers.videos import video_importer
from edps.task import TaskContext

Importer = Callable[[TaskContext, File], AsyncIterator[Analyzer]]

# Dictionary mapping a file extension to Importer
IMPORTERS: dict[str, Importer] = {
    "csv": csv_importer,
    "xls": xls_importer,
    "xlsx": xlsx_importer,
    "pdf": pdf_importer,
    "pickle": pickle_importer,
    "png": raster_image_importer,
    "jpg": raster_image_importer,
    "jpeg": raster_image_importer,
    "gif": raster_image_importer,
    "bmp": raster_image_importer,
    "tiff": raster_image_importer,
    "tif": raster_image_importer,
    "webp": raster_image_importer,
    "txt": unstructured_text_importer,
    "mp4": video_importer,
    "avi": video_importer,
    "mkv": video_importer,
    "mov": video_importer,
    "flv": video_importer,
    "wmv": video_importer,
}
