from pathlib import PurePosixPath

from extended_dataset_profile.models.v0.edp import CorrelationSummary
from pandas import DataFrame

from edps.analyzers.pandas import (
    _COMMON_UNIQUE,
    _get_correlation_graph,
    _get_correlation_matrix,
    _get_correlation_summary,
)


async def test_correlation(ctx):
    FIRST_COL = "elevator_accidents"
    SECOND_COL = "maintenance_interval"
    columns = DataFrame({FIRST_COL: [1, 2, 3, 4, 5], SECOND_COL: [5.0, 6.2, 7.5, 8.1, 9.9]})
    fields = DataFrame({_COMMON_UNIQUE: [5, 5]}, index=[FIRST_COL, SECOND_COL])
    correlation_matrix = await _get_correlation_matrix(
        ctx,
        columns,
        fields,
    )
    correlation_graph = await _get_correlation_graph(ctx, PurePosixPath("test_correlation_graph"), correlation_matrix)
    correlation_summary = await _get_correlation_summary(ctx, correlation_matrix)

    assert correlation_graph is not None
    assert correlation_summary == CorrelationSummary(no=0, partial=0, strong=1)
