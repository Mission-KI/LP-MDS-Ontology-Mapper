from pytest import raises

from edps.analyzers.unstructured_text.chunk import Chunk


def test_chunk_init():
    Chunk(0, 1)
    Chunk(0, 0)
    with raises(RuntimeError):
        Chunk(10, 1)


def test_chunk_overlap():
    base_chunk = Chunk(start_position_inclusive=0, end_position_exclusive=16)
    inside_chunk = Chunk(start_position_inclusive=9, end_position_exclusive=10)
    assert base_chunk.overlaps(inside_chunk)
    before_chunk = Chunk(start_position_inclusive=-10, end_position_exclusive=0)
    assert not base_chunk.overlaps(before_chunk)
    after_chunk = Chunk(start_position_inclusive=16, end_position_exclusive=17)
    assert not base_chunk.overlaps(after_chunk)


def test_position_setters():
    chunk = Chunk(start_position_inclusive=0, end_position_exclusive=16)
    with raises(RuntimeError):
        chunk.start_line_inclusive = 17
    with raises(RuntimeError):
        chunk.end_line_exclusive = -1
    chunk.start_line_inclusive = 1
    chunk.end_line_exclusive = 2


def test_chunk_subtract_non_overlapping():
    base_chunk = Chunk(start_position_inclusive=0, end_position_exclusive=16)
    non_contact_chunk = Chunk(start_position_inclusive=16, end_position_exclusive=17)
    result_chunks = list(base_chunk - non_contact_chunk)
    assert len(result_chunks) == 1
    assert result_chunks[0].start_line_inclusive == 0
    assert result_chunks[0].end_line_exclusive == 16
    # Check that base_chunk is unmodified.
    result_chunks[0].start_line_inclusive = 10
    assert base_chunk.start_line_inclusive == 0


def test_chunk_subtract_beginning():
    chunk = Chunk(start_position_inclusive=0, end_position_exclusive=16)
    subtract_chunk = Chunk(start_position_inclusive=0, end_position_exclusive=8)
    result_chunks = list(chunk - subtract_chunk)
    assert len(result_chunks) == 1
    assert result_chunks[0].start_line_inclusive == 8
    assert result_chunks[0].end_line_exclusive == 16


def test_chunk_subtract_end():
    chunk = Chunk(0, 16)
    subtract_chunk = Chunk(8, 16)
    result_chunks = list(chunk - subtract_chunk)
    assert len(result_chunks) == 1
    assert result_chunks[0].start_line_inclusive == 0
    assert result_chunks[0].end_line_exclusive == 8


def test_chunk_subtract_all():
    chunk = Chunk(0, 16)
    subtract_chunk = Chunk(0, 16)
    result_chunks = list(chunk - subtract_chunk)
    assert len(result_chunks) == 0


def test_chunk_subtract_middle():
    chunk = Chunk(0, 16)
    subtract_chunk = Chunk(4, 12)
    result_chunks = list(chunk - subtract_chunk)
    assert len(result_chunks) == 2
    assert result_chunks[0].start_line_inclusive == 0
    assert result_chunks[0].end_line_exclusive == 4
    assert result_chunks[1].start_line_inclusive == 12
    assert result_chunks[1].end_line_exclusive == 16
