from io import StringIO, TextIOBase
from itertools import permutations, tee
from pathlib import PurePosixPath
from re import compile
from typing import AsyncIterator, Generator, Iterable, Iterator, List, Optional, Tuple, Union
from uuid import UUID, uuid4
from warnings import warn

from clevercsv.cparser_util import parse_string
from clevercsv.dialect import SimpleDialect
from clevercsv.encoding import get_encoding
from clevercsv.exceptions import Error as CleverCsvError
from extended_dataset_profile.models.v0.edp import EmbeddedTable, StructuredDataSet, UnstructuredTextDataSet
from pandas import DataFrame

from edps.analyzers.base import Analyzer as _BaseAnalyzer
from edps.analyzers.pandas import PandasAnalyzer
from edps.analyzers.unstructured_text.chunk import Chunk, ChunkInterface
from edps.file import File
from edps.importers.structured import dialect_to_str, get_possible_csv_dialects
from edps.task import TaskContext


class Analyzer(_BaseAnalyzer):
    def __init__(self, file: File, parent_uuid: None | UUID = None, minimum_number_rows_csv: int = 3):
        self._file = file
        self._encoding = get_encoding(file.path)
        if self._encoding is None:
            raise RuntimeError("Could not determine encoding of %s", file)
        self._parent_uuid = parent_uuid
        self._uuid = uuid4()
        self._minimum_number_rows_csv = minimum_number_rows_csv
        self._word_splitter_regex = compile(r"\s+")

    async def analyze(self, ctx) -> AsyncIterator[Union[StructuredDataSet, UnstructuredTextDataSet]]:
        with open(self._file.path, "rt", encoding=self._encoding) as opened_file, StringIO() as unstructured_only_file:
            # unstructured_only_file the lines which where not handled by the CSV parsers.
            embedded_tables: List[EmbeddedTable] = []
            async for embbeded_table, csv_dataset in self._analyze_embedded_csv(
                ctx, opened_file, unstructured_only_file
            ):
                embedded_tables.append(embbeded_table)
                yield csv_dataset

            unstructured_only_file.seek(0)
            yield self._analyze_unstructured_text(unstructured_only_file, embedded_tables)

    def _analyze_unstructured_text(
        self, opened_file: TextIOBase, embedded_tables: List[EmbeddedTable]
    ) -> UnstructuredTextDataSet:
        line_count = 0
        word_count = 0
        for line in opened_file:
            line_count += 1
            word_count += self._count_words(line)

        return UnstructuredTextDataSet(
            uuid=self._uuid,
            parentUuid=self._parent_uuid,
            name=PurePosixPath(self._file.relative.as_posix()),
            embeddedTables=embedded_tables,
            lineCount=line_count,
            wordCount=word_count,
        )

    def _count_words(self, chunk: str) -> int:
        count = 0
        last_end_position = 0
        for match in self._word_splitter_regex.finditer(chunk):
            start_position, end_position = match.regs[0]
            current_chunk_size = start_position - last_end_position
            if current_chunk_size > 0:
                count += 1
            last_end_position = end_position
        return count

    async def _analyze_embedded_csv(
        self, ctx: TaskContext, opened_file: TextIOBase, remainder_buffer: TextIOBase
    ) -> AsyncIterator[Tuple[EmbeddedTable, StructuredDataSet]]:
        """This function extracts all CSV's and yields the readily analyzed StructuredDataSets.

        Details:
            It also outputs the remainder of the text (without CSV's) to the remainder_buffer.
        """
        with StringIO(newline="") as csv_buffer:
            analyzer = _CsvAnalyzer(
                ctx=ctx,
                opened_file=opened_file,
                csv_buffer=csv_buffer,
                remainder_buffer=remainder_buffer,
                minimum_number_rows=self._minimum_number_rows_csv,
                file=self._file,
                uuid=self._uuid,
            )
            async for result in analyzer.analyze(ctx):
                yield result


class _TrackedCsvChunk(ChunkInterface):
    def __init__(self, start_line_inclusive: int, column_count: int, dialect: SimpleDialect):
        super().__init__()
        self._start_line_inclusive = start_line_inclusive
        self._column_count = column_count
        self._dialect = dialect
        self._cells: List[List[str]] = []

    def append_row(self, row: List[str]) -> None:
        if len(row) != self._column_count:
            raise _LineParsingError(f"Can not append rows, expected {self._column_count} columns, got {len(row)}.")
        self._cells.append(row)

    @property
    def start_line_inclusive(self) -> int:
        return self._start_line_inclusive

    @property
    def end_line_exclusive(self) -> int:
        return self._start_line_inclusive + len(self._cells)

    @property
    def column_count(self) -> int:
        return self._column_count

    @property
    def dialect(self) -> SimpleDialect:
        return self._dialect

    @property
    def row_count(self) -> int:
        return len(self._cells)

    @property
    def cells(self) -> List[List[str]]:
        return self._cells

    def __sub__(self, other: "_TrackedCsvChunk") -> Iterator["_TrackedCsvChunk"]:
        chunk = Chunk(self.start_line_inclusive, self.end_line_exclusive)
        resulting_chunk_iterator = chunk - other
        for resulting_chunk in resulting_chunk_iterator:
            start_index = resulting_chunk.start_line_inclusive - self.start_line_inclusive
            end_index = resulting_chunk.end_line_exclusive - self.start_line_inclusive
            sub_chunk = _TrackedCsvChunk(
                start_line_inclusive=resulting_chunk.start_line_inclusive,
                column_count=self.column_count,
                dialect=self.dialect,
            )
            for row in self._cells[start_index:end_index]:
                sub_chunk.append_row(row)
            yield sub_chunk

    def __str__(self) -> str:
        return f"Chunk (lines: [{self.start_line_inclusive}, {self.end_line_exclusive}), dialect: {dialect_to_str(self.dialect.to_csv_dialect())})"


class _ChunkGroup(ChunkInterface):
    def __init__(self):
        self._chunks: List[_TrackedCsvChunk] = []

    def append(self, chunk: _TrackedCsvChunk) -> None:
        self._chunks.append(chunk)

    @property
    def start_line_inclusive(self) -> int:
        return min(chunk.start_line_inclusive for chunk in self._chunks)

    @property
    def end_line_exclusive(self) -> int:
        return max(chunk.end_line_exclusive for chunk in self._chunks)

    def empty(self) -> bool:
        return len(self._chunks) == 0

    def __len__(self) -> int:
        return len(self._chunks)

    def __iter__(self):
        return self._chunks.__iter__()

    def __str__(self) -> str:
        return f"group of chunks from line {self.start_line_inclusive} to {self.end_line_exclusive}"


class _CsvAnalyzer:
    def __init__(
        self,
        ctx: TaskContext,
        opened_file: TextIOBase,
        csv_buffer: TextIOBase,
        remainder_buffer: TextIOBase,
        minimum_number_rows: int,
        file: File,
        uuid: UUID,
    ):
        self._opened_file = opened_file
        self._csv_buffer = csv_buffer
        self.remainder_buffer = remainder_buffer
        self._parsers = _get_all_csv_parsers(ctx, minimum_number_rows, opened_file)
        self._minimum_number_rows = minimum_number_rows
        self._file = file
        self._uuid = uuid

    async def analyze(self, ctx: TaskContext) -> AsyncIterator[Tuple[EmbeddedTable, StructuredDataSet]]:
        chunks = [chunk async for chunk in self._walk_file()]
        grouped_chunks = list(self._group_overlapping_chunks(chunks))
        chunks = [chunk for group in grouped_chunks for chunk in self._resolve_overlaps(ctx, group)]
        number_lines = self._count_lines()
        unstructured_text_chunks = [Chunk(0, number_lines + 1)]

        for structured_chunk in chunks:
            ctx.logger.info(
                "Analyzing lines %d to %d as csv with dialect (%s).",
                structured_chunk.start_line_inclusive,
                structured_chunk.end_line_exclusive,
                dialect_to_str(structured_chunk.dialect.to_csv_dialect()),
            )
            analyzer = PandasAnalyzer(
                DataFrame(structured_chunk.cells[1:], dtype=str, columns=structured_chunk.cells[0]), self._file
            )
            async for dataset in ctx.exec(analyzer.analyze):
                dataset.parentUuid = self._uuid
                embedded_table = EmbeddedTable(
                    startLine=structured_chunk.start_line_inclusive,
                    endLine=structured_chunk.end_line_exclusive,
                    structuredDatasetUuid=dataset.uuid,
                )
                yield embedded_table, dataset
            # Subtract structured chunk from unstructured chunks
            unstructured_text_chunks = [
                result_chunk
                for unstructured_chunk in unstructured_text_chunks
                for result_chunk in unstructured_chunk - structured_chunk
            ]
        self._opened_file.seek(0)
        for line_index, line_text in enumerate(self._opened_file):
            if any(line_index in unstructured_chunk for unstructured_chunk in unstructured_text_chunks):
                self.remainder_buffer.write(line_text)

    async def _walk_file(self) -> AsyncIterator[_TrackedCsvChunk]:
        self._opened_file.seek(0)
        for index, text in enumerate(self._opened_file):
            for parser in self._parsers:
                async for chunk in parser.feed_line(index, text):
                    yield chunk

        for parser in self._parsers:
            async for chunk in parser.end_of_input():
                yield chunk

    def _group_overlapping_chunks(self, chunks: List[_TrackedCsvChunk]) -> Iterator[_ChunkGroup]:
        """Groups chunks that are overlapping together."""
        by_start = sorted(chunks, key=lambda chunk: chunk.start_line_inclusive)
        group = _ChunkGroup()
        for chunk in by_start:
            if group.empty() or group.overlaps(chunk):
                group._chunks.append(chunk)
            else:
                yield group
                group = _ChunkGroup()
                group._chunks.append(chunk)
        if not group.empty():
            yield group

    def _resolve_overlaps(self, ctx: TaskContext, overlap_grouped_chunks: _ChunkGroup) -> List[_TrackedCsvChunk]:
        """Resolves each group of overlaps to maximize the number of parsed lines."""
        # Walk through all possible permutations of the group.
        # In all permutations, we use first come first serve for chunks.
        # That should give us all possible combinations.
        if len(overlap_grouped_chunks) == 1:
            return list(overlap_grouped_chunks)
        ctx.logger.debug("Trying to resolve overlapping %s.", overlap_grouped_chunks)
        best_permutation_so_far: Optional[List[_TrackedCsvChunk]] = None
        most_number_of_lines_parsed = 0
        for permuted_chunks in permutations(overlap_grouped_chunks._chunks):
            exclusive_chunks = self._build_exclusive_chunk_group(permuted_chunks)
            number_of_lines_parsed = sum(chunk.row_count for chunk in exclusive_chunks)
            if number_of_lines_parsed > most_number_of_lines_parsed:
                best_permutation_so_far = exclusive_chunks
                most_number_of_lines_parsed = number_of_lines_parsed
        if best_permutation_so_far is None:
            raise RuntimeError("Could not resolve overlaps for the chunks.")
        ctx.logger.debug(
            "Resolved overlaps in %s to [%s].",
            overlap_grouped_chunks,
            ", ".join(str(chunk) for chunk in best_permutation_so_far),
        )
        return best_permutation_so_far

    def _build_exclusive_chunk_group(self, chunks: Iterable[_TrackedCsvChunk]) -> List[_TrackedCsvChunk]:
        exclusive_chunks: List[_TrackedCsvChunk] = []
        for chunk in chunks:
            for sanitized_chunk in self._subtract_chunks(chunk, iter(exclusive_chunks)):
                exclusive_chunks.append(sanitized_chunk)
        return exclusive_chunks

    def _subtract_chunks(
        self, chunk: _TrackedCsvChunk, subtract_chunks_iterator: Iterator[_TrackedCsvChunk]
    ) -> Generator[_TrackedCsvChunk, None, None]:
        try:
            next_subtract_chunk = next(subtract_chunks_iterator)
        except StopIteration:
            next_subtract_chunk = None

        if next_subtract_chunk is None:
            # No next subtract chunk. Just return chunk chunk.
            yield chunk
        else:
            sub_chunks = list(chunk - next_subtract_chunk)
            teed_subtract_iterators = tee(subtract_chunks_iterator, len(sub_chunks))
            for sub_chunk, teed_subtract_chunks_iterator in zip(sub_chunks, teed_subtract_iterators):
                yield from self._subtract_chunks(sub_chunk, teed_subtract_chunks_iterator)

    def _count_lines(self):
        start_position = self._opened_file.tell()
        self._opened_file.seek(0)
        line_count = sum(1 for line in self._opened_file)
        self._opened_file.seek(start_position)
        return line_count


def _get_all_csv_parsers(ctx: TaskContext, minimum_number_rows: int, opened_file: TextIOBase):
    return [
        _CsvParser(ctx, dialect, minimum_number_rows)
        for dialect in get_possible_csv_dialects(opened_file)
        if dialect.delimiter != " "
    ]


class _LineParsingError(RuntimeError):
    """This exception is thrown, when a parser fails to parse a single line."""


class _CsvParser:
    def __init__(self, ctx: TaskContext, dialect: SimpleDialect, minimum_number_rows: int):
        super().__init__()
        self.minimum_number_rows = minimum_number_rows
        self.dialect = dialect
        self._tracked_chunk: Optional[_TrackedCsvChunk] = None
        self._ctx = ctx

    async def feed_line(self, line_index: int, line_text: str) -> AsyncIterator[_TrackedCsvChunk]:
        try:
            line_elements = self._parse_csv_line(line_text)
            if self._tracked_chunk is None:
                self._tracked_chunk = _TrackedCsvChunk(
                    start_line_inclusive=line_index, dialect=self.dialect, column_count=len(line_elements)
                )
            self._tracked_chunk.cells.append(line_elements)
        except _LineParsingError:
            async for tracked_chunk in self.end_of_input():
                yield tracked_chunk

    async def end_of_input(self) -> AsyncIterator[_TrackedCsvChunk]:
        if self._tracked_chunk and (self._tracked_chunk.row_count >= self.minimum_number_rows):
            yield self._tracked_chunk
        self._tracked_chunk = None

    def _parse_csv_line(self, line: str) -> List[str]:
        try:
            row_iterator = parse_string(line, self.dialect)
        except CleverCsvError as error:
            # This dialect can not be used on the given row.
            raise _LineParsingError() from error
        row = next(row_iterator)
        if len(row) <= 1:
            raise _LineParsingError()

        try:
            next(row_iterator)
            message = "SimpleCsv parsed multiple rows, expected only one!"
            warn(message)
            self._ctx.logger.warning(message)
            raise _LineParsingError()
        except StopIteration:
            pass

        if self._tracked_chunk and (self._tracked_chunk.column_count != len(row)):
            raise _LineParsingError("Number of columns does not match!")
        return row
