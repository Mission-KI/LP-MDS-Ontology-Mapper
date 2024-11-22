from asyncio import get_running_loop
from csv import Dialect
from logging import Logger, getLogger

from clevercsv import Detector
from clevercsv.detect import DetectionMethod
from clevercsv.encoding import get_encoding
from pandas import read_csv

from edp.analyzers.pandas import Pandas
from edp.file import File


async def csv(file: File):
    logger = getLogger("CSV Importer")
    logger.info("Importing %s as CSV", file)

    dialect, encoding, has_header = _detect_dialect_encoding_and_has_header(logger, file, num_chars=5000)

    csv_dialect: Dialect | None = None
    if dialect:
        csv_dialect = dialect.to_csv_dialect()
        logger.info(
            "Detected dialect. Delimiter=%s, EscapeChar=%s, QuoteChar=%s",
            dialect.delimiter,
            dialect.escapechar,
            dialect.quotechar,
        )
    else:
        logger.info("No dialect detected")

    logger.info("%s", "Detected Header" if has_header else "No Header detected")

    def runner():
        return read_csv(
            file.path, dialect=csv_dialect, header="infer" if has_header else None, encoding=encoding  # type: ignore
        )

    loop = get_running_loop()
    data_frame = await loop.run_in_executor(None, runner)

    # Set column headers (col000, col001, ...) if the csv doesn't contain headers.
    # Otherwise the headers are just numbers, not strings.
    if not has_header:
        col_count = len(data_frame.columns)
        data_frame.columns = [f"col{i:03d}" for i in range(col_count)]

    return Pandas(data_frame, file)


def _detect_dialect_encoding_and_has_header(logger: Logger, file: File, num_chars: int):
    enc = get_encoding(file.path)
    if enc is None:
        raise RuntimeError(f'Could not detect encoding for "{file.relative}"')

    logger.info('Detected encoding "%s"', enc)
    # newline="" is important for CSV files as it preserves line endings.
    # Thus read() passes them through to the detector.
    with file.path.open("r", newline="", encoding=enc) as fp:
        data = fp.read(num_chars) if num_chars else fp.read()
        detector = Detector()
        dialect = detector.detect(data, verbose=False, method=DetectionMethod.AUTO)
        has_header = detector.has_header(data)
    return dialect, enc, has_header
