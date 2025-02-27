from pathlib import Path

from extended_dataset_profile.models.v0.edp import Resolution, VideoCodec, VideoDataSet, VideoPixelFormat

from edps.file import File
from edps.importers import lookup_importer


async def test_import_mp4(path_data_test_mp4, ctx):
    file = get_file(path_data_test_mp4)
    video_importer = lookup_importer("mp4")
    await ctx.exec("dataset", video_importer, file)
    dataset = next(ctx.collect_datasets())
    assert isinstance(dataset, VideoDataSet)
    assert dataset.codec == VideoCodec.H264
    assert dataset.resolution == Resolution(width=1280, height=720)
    assert abs(dataset.fps - 30) < 0.1
    assert abs(dataset.duration - 30.5) < 0.1
    assert dataset.pixel_format == VideoPixelFormat.YUV420P


def get_file(path: Path):
    return File(path.parent, path)
