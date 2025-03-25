from pathlib import Path
from warnings import warn

import ffmpeg
import static_ffmpeg
from extended_dataset_profile.models.v0.edp import Resolution, VideoCodec, VideoDataSet, VideoPixelFormat

from edps.analyzers.videos import VideoAnalyzer, VideoMetadata
from edps.taskcontext import TaskContext


async def video_importer(ctx: TaskContext, path: Path) -> VideoDataSet:
    ctx.logger.info("Analyzing video '%s'", ctx.relative_path(path))

    # Blocks until files are downloaded, but only if ffmpeg not already on path
    static_ffmpeg.add_paths(weak=True)

    probe = ffmpeg.probe(path)
    video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]

    if not video_streams:
        raise RuntimeError(f'Could not detect video streams for "{ctx.relative_path(path)}"')

    video_stream = video_streams[0]
    codec = video_stream.get("codec_name", "UNKNOWN")
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    fps = _compute_fps(ctx, video_stream.get("avg_frame_rate", "0/1"))
    duration = float(probe.get("format", {}).get("duration", "0.0"))
    pixel_format = video_stream.get("pix_fmt", "UNKNOWN")

    metadata = VideoMetadata(
        codec=VideoCodec(codec),
        resolution=Resolution(width=width, height=height),
        fps=fps,
        duration=duration,
        pixel_format=VideoPixelFormat(pixel_format),
    )
    analyzer = VideoAnalyzer(metadata)
    return await analyzer.analyze(ctx)


def _compute_fps(ctx: TaskContext, avg_frame_rate: str) -> float:
    try:
        num, den = avg_frame_rate.split("/")
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError) as error:
        message = f"Could not determine video FPS: {error}"
        ctx.logger.warning(message)
        warn(message)
        return 0.0
