from pathlib import Path

from extended_dataset_profile.models.v0.edp import Resolution, VideoCodec, VideoPixelFormat

from edps.file import File
from edps.importers import lookup_importer


async def test_import_mp4(path_data_test_mp4, ctx):
    file = get_file(path_data_test_mp4)
    video_importer = lookup_importer("mp4")
    analyzer = await anext(ctx.exec(video_importer, file))
    metadata = analyzer._metadata
    assert metadata.codec == VideoCodec.H264
    assert metadata.resolution == Resolution(width=1280, height=720)
    assert abs(metadata.fps - 30) < 0.1
    assert abs(metadata.duration - 30.5) < 0.1
    assert metadata.pixel_format == VideoPixelFormat.YUV420P


def get_file(path: Path):
    return File(path.parent, path)
