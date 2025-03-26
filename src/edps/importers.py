from pathlib import Path
from typing import Awaitable, Callable, Optional

from edps.analyzers.archive import archive_importer
from edps.analyzers.docx import docx_importer
from edps.analyzers.images.importer import raster_image_importer
from edps.analyzers.media import media_importer
from edps.analyzers.pdf import pdf_importer
from edps.analyzers.python import pickle_importer
from edps.analyzers.semi_structured.importer import json_importer
from edps.analyzers.structured.importer import csv_importer, xls_importer, xlsx_importer
from edps.analyzers.unstructured_text.importer import unstructured_text_importer
from edps.taskcontext import TaskContext
from edps.types import DataSet

Importer = Callable[[TaskContext, Path], Awaitable[DataSet]]


# Dictionary mapping a file type (extension) to an Importer.
_IMPORTERS: dict[str, Importer] = {
    "csv": csv_importer,
    "xls": xls_importer,
    "xlsx": xlsx_importer,
    "pdf": pdf_importer,
    "docx": docx_importer,
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
    "json": json_importer,
    "avi": media_importer,
    "flv": media_importer,
    "mkv": media_importer,
    "mov": media_importer,
    "mp4": media_importer,
    "wmv": media_importer,
    "aac": media_importer,
    "flac": media_importer,
    "m4a": media_importer,
    "mp3": media_importer,
    "ogg": media_importer,
    "opus": media_importer,
    "wav": media_importer,
    "wma": media_importer,
    "zip": archive_importer,
}


# Dictionary of unsupported types with specific message.
_UNSUPPORTED_TYPE_MESSAGES: dict[str, str] = {
    "doc": "DOC format is not supported directly. Please convert it to DOCX or PDF manually.",
}


def get_importable_types() -> str:
    return ", ".join(_IMPORTERS)


def lookup_importer(file_type: str) -> Optional[Importer]:
    if file_type in _IMPORTERS:
        return _IMPORTERS[file_type]
    return None


def lookup_unsupported_type_message(file_type: str) -> str:
    if file_type in _UNSUPPORTED_TYPE_MESSAGES:
        return _UNSUPPORTED_TYPE_MESSAGES[file_type]
    return f'Import for "{file_type}" not yet supported'
