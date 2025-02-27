from asyncio import get_running_loop
from pathlib import PurePosixPath

import numpy as np
from extended_dataset_profile.models.v0.edp import ImageColorMode, ImageDataSet, ImageDPI, Resolution
from PIL.Image import Image
from PIL.Image import open as open_image

from edps.analyzers.images import ImageAnalyzer, ImageMetadata
from edps.file import File
from edps.task import TaskContext


async def raster_image_importer(ctx: TaskContext, file: File) -> ImageDataSet:
    ctx.logger.info("Analyzing raster image '%s'", file)

    def runner():
        with open_image(file.path) as img:
            return parse_raster_image(img)

    img_metadata, img_array = await get_running_loop().run_in_executor(None, runner)
    name = PurePosixPath(file.relative)
    analyzer = ImageAnalyzer(img_metadata, img_array, name)
    return await analyzer.analyze(ctx)


async def raster_image_importer_from_pilimage(ctx: TaskContext, image: Image) -> ImageDataSet:
    ctx.logger.info("Analyzing raster image '%s'", ctx.dataset_name)
    img_metadata, img_array = parse_raster_image(image)
    name = PurePosixPath(ctx.dataset_name or "UNKNOWN")
    analyzer = ImageAnalyzer(img_metadata, img_array, name)
    return await analyzer.analyze(ctx)


def parse_raster_image(img: Image) -> tuple[ImageMetadata, np.ndarray]:
    codec = img.format
    color_mode = img.mode
    resolution = img.size
    dpi = img.info.get("dpi", (0.0, 0.0))
    img_rgb = img.convert(ImageColorMode.RGB)
    img_array = np.array(img_rgb)
    img_metadata = ImageMetadata(
        codec=codec or "UNKNOWN",
        color_mode=ImageColorMode(color_mode),
        resolution=Resolution(width=resolution[0], height=resolution[1]),
        dpi=ImageDPI(x=dpi[0], y=dpi[1]),
    )
    return img_metadata, img_array
