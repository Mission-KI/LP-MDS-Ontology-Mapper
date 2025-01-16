from pandas import DataFrame

from edps.analyzers.pandas import _COMMON_UNIQUE, _get_correlation_graph


async def test_correlation(ctx):
    FIRST_COL = "elevator_accidents"
    SECOND_COL = "maintenance_interval"
    columns = DataFrame({FIRST_COL: [1, 2, 3, 4, 5], SECOND_COL: [5.0, 6.2, 7.5, 8.1, 9.9]})
    fields = DataFrame({_COMMON_UNIQUE: [5, 5]}, index=[FIRST_COL, SECOND_COL])
    await _get_correlation_graph(ctx, "test_correlation_graph", columns, fields)
