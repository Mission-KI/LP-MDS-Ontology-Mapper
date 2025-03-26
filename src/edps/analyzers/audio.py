import io
from pathlib import Path, PurePosixPath
from typing import Any

from matplotlib.figure import figaspect
import numpy as np
import ffmpeg
from numpy import ndarray
import scipy
import static_ffmpeg
from extended_dataset_profile.models.v0.edp import AudioDataSet

from edps.filewriter import get_pyplot_writer
from edps.taskcontext import TaskContext


async def audio_importer(ctx: TaskContext, path: Path) -> AudioDataSet:
    ctx.logger.info("Analyzing audio '%s'", ctx.relative_path(path))

    # Blocks until files are downloaded, but only if ffmpeg not already on path
    static_ffmpeg.add_paths(weak=True)

    probe = ffmpeg.probe(path)

    audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]

    if not audio_streams:
        raise RuntimeError(f'Could not detect audio streams for "{ctx.relative_path(path)}"')

    return await analyse_audio(ctx, path, audio_streams[0], 0)


async def analyse_audio(ctx: TaskContext, path: Path, audio_stream: Any, stream_number: int) -> AudioDataSet:
    codec = audio_stream.get("codec_name", "UNKNOWN")
    channels = audio_stream.get("channels", 0)
    duration = float(audio_stream.get("duration", "0.0"))
    sample_rate = int(audio_stream.get("sample_rate", "0"))
    bit_rate = int(audio_stream.get("bit_rate", "0"))
    bits_per_sample = int(audio_stream.get("bits_per_sample", "0"))

    spectrogram_ref = await plot_spectrogram(ctx, path, stream_number)

    return AudioDataSet(
        codec=codec,
        channels=channels,
        duration=duration,
        sampleRate=sample_rate,
        bitRate=bit_rate,
        bitsPerSample=bits_per_sample if bits_per_sample != 0 else None,
        spectrogram=spectrogram_ref,
    )


async def plot_spectrogram(ctx: TaskContext, path: Path, stream_number: int) -> PurePosixPath:
    sampling_rate, samples = load_audio_samples(path, stream_number)

    LOWEST_FREQ = 20     # 20 Hz
    MIN_RANGE = 1e-1     # limit range to -10 dB

    window_size = int(2 * sampling_rate / LOWEST_FREQ)
    n = len(samples)

    window = scipy.signal.windows.gaussian(window_size, 0.25 * window_size, sym=True)
    sfft = scipy.signal.ShortTimeFFT(
        window, int(0.1 * window_size), fs=sampling_rate, fft_mode="onesided"
    )
    spectrogram = sfft.spectrogram(samples, padding="odd")
    spectrogram_log = 10 * np.log10(np.fmax(spectrogram, MIN_RANGE))

    plot_name = ctx.build_output_reference("fft")
    async with get_pyplot_writer(ctx, plot_name, figsize=figaspect(0.5)) as (axes, reference):
        t_lo, t_hi = sfft.extent(n)[:2]     # time range of plot
        axes.set_title(f"Spectrogram for {ctx.dataset_name}")
        axes.set(xlabel=f"Time $t$ in seconds " + 
                    rf"({sfft.p_num(n)} slices, $\Delta t = {sfft.delta_t:g}\,$s)",
                ylabel=f"Freq. $f$ in Hz " + 
                    rf"({sfft.f_pts} bins, $\Delta f = {sfft.delta_f:g}\,$Hz)",
                xlim=(t_lo, t_hi))
        axes.imshow(spectrogram_log, origin='lower', aspect="auto", extent=sfft.extent(n), cmap='magma')
        return reference


def load_audio_samples(path: Path, audio_stream: int) -> tuple[int, ndarray]:
    # Convert to mono WAV format in memory, take nth audio stream.
    stream_mapping = f"0:a:{audio_stream}"
    wav_data, _ = (ffmpeg
        .input(path)
        .output("pipe:", format="wav", acodec="pcm_s16le", ar=44100, ac=1, map=stream_mapping)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

    # RIFF WAV files have an 8 bytes header: marker "RIFF", then content size as 4 bytes in little endian.
    # For ffmpeg piping output the content size is missing. So we fix these 4 bytes.
    size_without_header = len(wav_data) - 8
    wav_data_io = io.BytesIO(wav_data)
    wav_data_io.seek(4)
    size_bytes = int.to_bytes(size_without_header, length=4, byteorder="little")
    wav_data_io.write(size_bytes)
    wav_data_io.seek(0)

    # Read WAV data using SciPy
    sampling_rate, samples = scipy.io.wavfile.read(wav_data_io)
    return sampling_rate, samples
