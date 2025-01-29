from pathlib import Path

from extended_dataset_profile.models.v0.edp import ImageColorMode, ImageDimensions, ImageDPI

from edps.file import File
from edps.importers.images import raster_image_importer


async def test_import_png(path_data_test_png, ctx):
    file = get_file(path_data_test_png)
    analyzer = await anext(ctx.exec(raster_image_importer, file))
    metadata = analyzer._metadata
    data = analyzer._data
    height, width, _ = data.shape
    assert metadata.codec == "PNG"
    assert metadata.color_mode == ImageColorMode.PALETTED
    assert metadata.resolution == ImageDimensions(width=600, height=400)
    assert metadata.dpi == ImageDPI(x=96.012, y=96.012)
    assert height == 400
    assert width == 600


async def test_import_jpg(path_data_test_jpg, ctx):
    file = get_file(path_data_test_jpg)
    analyzer = await anext(ctx.exec(raster_image_importer, file))
    metadata = analyzer._metadata
    data = analyzer._data
    height, width, _ = data.shape
    assert metadata.codec == "JPEG"
    assert metadata.color_mode == ImageColorMode.RGB
    assert metadata.resolution == ImageDimensions(width=600, height=400)
    assert metadata.dpi == ImageDPI(x=96, y=96)
    assert height == 400
    assert width == 600


def get_file(path: Path):
    return File(path.parent, path)
