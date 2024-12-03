from logging import getLogger

from pandas import DataFrame

from edp.analyzers.pandas import _COMMON_UNIQUE, _get_correlation_graph


async def test_correlation(output_context):
    FIRST_COL = "elevator_accidents"
    SECOND_COL = "maintenance_interval"
    columns = DataFrame({FIRST_COL: [1, 2, 3, 4, 5], SECOND_COL: [5.0, 6.2, 7.5, 8.1, 9.9]})
    fields = DataFrame({_COMMON_UNIQUE: [5, 5]}, index=[FIRST_COL, SECOND_COL])
    await _get_correlation_graph(getLogger("Test"), "test_correlation_graph", columns, fields, output_context)
