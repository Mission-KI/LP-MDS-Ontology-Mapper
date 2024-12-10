from asyncio import get_running_loop
from csv import Dialect
from logging import Logger, getLogger
from typing import Literal

from clevercsv import Detector
from clevercsv.detect import DetectionMethod
from clevercsv.encoding import get_encoding
from pandas import read_csv, read_excel

from edp.analyzers.pandas import Pandas
from edp.file import File


async def csv_importer(file: File):
    logger = getLogger("CSV Importer")
    logger.info("Importing '%s' as CSV", file)

    dialect, encoding, has_header = _detect_dialect_encoding_and_has_header(logger, file, num_lines=100)

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
            file.path,
            dialect=csv_dialect,  # type: ignore
            header="infer" if has_header else None,
            encoding=encoding,
        )

    loop = get_running_loop()
    data_frame = await loop.run_in_executor(None, runner)

    # Set column headers (col000, col001, ...) if the csv doesn't contain headers.
    # Otherwise the headers are just numbers, not strings.
    if not has_header:
        col_count = len(data_frame.columns)
        data_frame.columns = [f"col{i:03d}" for i in range(col_count)]

    return Pandas(data_frame, file)


async def excel_importer(file: File, engine: Literal["xlrd", "openpyxl"]):
    """Import XLS/XLSX files. The engine must be passed explicitly to ensure the required libraries are installed."""

    logger = getLogger("Excel Importer")
    logger.info("Importing '%s' as Excel", file)

    def runner():
        return read_excel(
            file.path,
            engine=engine,
            sheet_name=None,  # imports all sheets as dictionary
            header=0,  # assume header in row 0
        )

    loop = get_running_loop()
    dataframes_map = await loop.run_in_executor(None, runner)

    sheets = list(dataframes_map.keys())
    dataframes = list(dataframes_map.values())

    # TODO: Importer should be able to provide multiple outputs (DataFrames) in the future!
    if len(sheets) == 0:
        raise RuntimeError("Excel contains no sheets!")
    if len(sheets) == 1:
        logger.info("Data imported from sheet '%s'", sheets[0])
    if len(sheets) > 1:
        logger.warning("Excel contains multiple sheets %s, only first one is used!", sheets)

    return Pandas(dataframes[0], file)


async def xlsx_importer(file: File):
    return await excel_importer(file, "openpyxl")


async def xls_importer(file: File):
    return await excel_importer(file, "xlrd")


def _detect_dialect_encoding_and_has_header(logger: Logger, file: File, num_lines: int | None):
    enc = get_encoding(file.path)
    if enc is None:
        raise RuntimeError(f'Could not detect encoding for "{file.relative}"')
    logger.info('Detected encoding "%s"', enc)

    data = _read_file_top_lines(file, enc, num_lines)

    detector = Detector()
    dialect = detector.detect(data, verbose=False, method=DetectionMethod.AUTO)
    has_header = detector.has_header(data)
    return dialect, enc, has_header


def _read_file_top_lines(file: File, enc: str, num_lines: int | None) -> str:
    data: str = ""
    # newline="" is important for CSV files as it preserves line endings.
    # Thus read() passes them through to the detector.
    with file.path.open("r", newline="", encoding=enc) as fp:
        for index, line in enumerate(fp):
            if not num_lines or index < num_lines:
                data += line
            else:
                break
    return data
