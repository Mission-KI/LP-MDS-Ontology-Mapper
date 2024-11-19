from asyncio import get_running_loop
from logging import getLogger

from clevercsv import Detector
from clevercsv.detect import DetectionMethod
from clevercsv.encoding import get_encoding
from pandas import read_csv

from edp.analyzers.pandas import Pandas
from edp.file import File


async def csv(file: File):
    logger = getLogger("CSV Importer")
    logger.info("Importing %s as CSV", file)

    dialect, has_header = _detect_dialect_and_has_header(file, num_chars=5000)

    def runner():
        return read_csv(
            file.path,
            dialect=dialect.to_csv_dialect() if dialect is not None else None,  # type: ignore
            header="infer" if has_header else None,
        )

    loop = get_running_loop()
    data_frame = await loop.run_in_executor(None, runner)

    # Set column headers (col000, col001, ...) if the csv doesn't contain headers.
    # Otherwise the headers are just numbers, not strings.
    if not has_header:
        col_count = len(data_frame.columns)
        data_frame.columns = [f"col{i:03d}" for i in range(col_count)]

    return Pandas(data_frame, file)


def _detect_dialect_and_has_header(file: File, num_chars: int):
    enc = get_encoding(file.path)
    # newline="" is important for CSV files as it preserves line endings.
    # Thus read() passes them through to the detector.
    with file.path.open("r", newline="", encoding=enc) as fp:
        data = fp.read(num_chars) if num_chars else fp.read()
        detector = Detector()
        dialect = detector.detect(data, verbose=False, method=DetectionMethod.AUTO)
        has_header = detector.has_header(data)
    return dialect, has_header
