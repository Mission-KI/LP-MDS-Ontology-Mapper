import warnings
from pathlib import Path
from typing import Awaitable, Callable, Optional

from extended_dataset_profile.models.v0.edp import DataSet

from edps.file import determine_file_type
from edps.importers.docx import docx_importer
from edps.importers.images import raster_image_importer
from edps.importers.pdf import pdf_importer
from edps.importers.python import pickle_importer
from edps.importers.structured import csv_importer, xls_importer, xlsx_importer
from edps.importers.unstructured_text import unstructured_text_importer
from edps.importers.videos import video_importer
from edps.task import TaskContext

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
    "mp4": video_importer,
    "avi": video_importer,
    "mkv": video_importer,
    "mov": video_importer,
    "flv": video_importer,
    "wmv": video_importer,
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


async def import_file(ctx: TaskContext, dataset_name: str, path: Path) -> None:
    """Import and analyze the file if it's a supported type. Yield calculated datasets."""
    file_type = determine_file_type(path)
    importer = lookup_importer(file_type)
    if importer is None:
        message = lookup_unsupported_type_message(file_type)
        ctx.logger.warning(message)
        warnings.warn(message, RuntimeWarning)
    else:
        await ctx.exec(dataset_name, importer, path)
