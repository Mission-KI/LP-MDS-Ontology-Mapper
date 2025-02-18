from asyncio import get_running_loop
from csv import Dialect
from io import TextIOBase
from typing import AsyncIterator, Literal

from clevercsv import Detector
from clevercsv.detect import DetectionMethod
from clevercsv.dialect import SimpleDialect
from clevercsv.encoding import get_encoding
from clevercsv.potential_dialects import get_dialects as _get_dialects
from pandas import Index, read_csv, read_excel

from edps.analyzers import PandasAnalyzer
from edps.file import File
from edps.task import TaskContext


async def csv_importer(ctx: TaskContext, file: File) -> AsyncIterator[PandasAnalyzer]:
    ctx.logger.info("Importing '%s' as CSV", file)

    dialect, encoding, has_header = _detect_dialect_encoding_and_has_header(ctx, file, num_lines=100)
    csv_dialect = _translate_dialect(ctx, dialect)
    ctx.logger.info("%s", "Detected Header" if has_header else "No Header detected")

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
        data_frame.columns = Index([f"col{i:03d}" for i in range(col_count)])

    yield PandasAnalyzer(data_frame, file)


def _translate_dialect(ctx, dialect: None | SimpleDialect) -> None | Dialect:
    if dialect:
        converted_dialect = dialect.to_csv_dialect()
        ctx.logger.info("Detected dialect: %s", dialect_to_str(converted_dialect))
        return converted_dialect
    else:
        ctx.logger.info("No dialect detected")
        return None


async def excel_importer(
    ctx: TaskContext, file: File, engine: Literal["xlrd", "openpyxl"]
) -> AsyncIterator[PandasAnalyzer]:
    """Import XLS/XLSX files. The engine must be passed explicitly to ensure the required libraries are installed."""

    ctx.logger.info("Importing '%s' as Excel", file)

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
        ctx.logger.info("Data imported from sheet '%s'", sheets[0])
    if len(sheets) > 1:
        ctx.logger.warning("Excel contains multiple sheets %s, only first one is used!", sheets)

    yield PandasAnalyzer(dataframes[0], file)


def xlsx_importer(ctx: TaskContext, file: File) -> AsyncIterator[PandasAnalyzer]:
    return excel_importer(ctx, file, "openpyxl")


def xls_importer(ctx: TaskContext, file: File) -> AsyncIterator[PandasAnalyzer]:
    return excel_importer(ctx, file, "xlrd")


def get_possible_csv_dialects(opened_file: TextIOBase):
    data = opened_file.read()
    encoding = opened_file.encoding
    return _get_dialects(data, encoding=encoding)


def dialect_to_str(dialect: Dialect) -> str:
    text = ""
    text += f'delimiter "{dialect.delimiter}", ' if dialect.delimiter else "no delimiter, "
    text += f'escapechar "{dialect.escapechar}", ' if dialect.escapechar else "no escapechar, "
    text += f'quotechar "{dialect.quotechar}"' if dialect.quotechar else "no quotechar"
    return text


def _detect_dialect_encoding_and_has_header(ctx: TaskContext, file: File, num_lines: int | None):
    enc = get_encoding(file.path)
    if enc is None:
        raise RuntimeError(f'Could not detect encoding for "{file.relative}"')
    ctx.logger.info('Detected encoding "%s"', enc)

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
