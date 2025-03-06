from typing import List, Optional
from warnings import warn

from extended_dataset_profile.models.v0.edp import TemporalConsistency
from numpy import count_nonzero
from pandas import DataFrame, Series, Timedelta, UInt64Dtype

from edps.taskcontext import TaskContext


class DatetimeColumnTemporalConsistency:
    def __init__(
        self,
        period: str,
        temporal_consistencies: List[TemporalConsistency],
        cleaned_series: Series,
    ) -> None:
        self.period = period
        self.temporal_consistencies = temporal_consistencies
        self.cleaned_series = cleaned_series

    def get_main_temporal_consistency(self) -> TemporalConsistency:
        return self.__getitem__(self.period)

    def __getitem__(self, period: str) -> TemporalConsistency:
        try:
            return next((consistency for consistency in self.temporal_consistencies if consistency.timeScale == period))
        except StopIteration as error:
            raise KeyError(f"{period} not in the temporal consistencies") from error


async def compute_temporal_consistency(ctx: TaskContext, columns: DataFrame) -> Series:
    return Series({name: _compute_temporal_consistency_for_column(ctx, column) for name, column in columns.items()})


def determine_periodicity(gaps: Series, distincts: Series) -> str:
    diff = distincts - gaps
    return str(diff.idxmax())


def _compute_temporal_consistency_for_column(
    ctx: TaskContext, column: Series
) -> Optional[DatetimeColumnTemporalConsistency]:
    column = column.sort_values(ascending=True)
    row_count = len(column)

    TIME_BASE_THRESHOLD = 15
    unique_timestamps = column.nunique()
    if unique_timestamps < TIME_BASE_THRESHOLD:
        message = (
            "Can not analyze temporal consistency, time base contains too few unique timestamps. "
            f"Have {unique_timestamps}, need at least {TIME_BASE_THRESHOLD}."
        )
        ctx.logger.warning(message)
        warn(message)
        return None

    # Remove null entries
    column = column[column.notnull()]  # type: ignore
    new_count = len(column)
    if new_count < row_count:
        empty_index_count = row_count - new_count
        message = f"Filtered out {empty_index_count} rows, because their index was empty"
        ctx.logger.warning(message)
        warn(message)
        row_count = new_count

    granularities = {
        "microseconds": Timedelta(microseconds=1),
        "milliseconds": Timedelta(milliseconds=1),
        "seconds": Timedelta(seconds=1),
        "minutes": Timedelta(minutes=1),
        "hours": Timedelta(hours=1),
        "days": Timedelta(days=1),
        "weeks": Timedelta(weeks=1),
        "months": Timedelta(days=30),
        "years": Timedelta(days=365),
    }

    deltas = column.diff()[1:]
    gaps = Series(
        {label: count_nonzero(deltas > time_base) for label, time_base in granularities.items()},
        dtype=UInt64Dtype(),
    )
    distincts = Series(
        {label: column.dt.round(time_base).nunique() for label, time_base in granularities.items()},  # type: ignore
        dtype=UInt64Dtype(),
    )
    stable = distincts == 1
    periodicity = determine_periodicity(gaps, distincts)
    temporal_consistencies = [
        TemporalConsistency(
            timeScale=time_base,
            differentAbundancies=distincts[time_base],
            stable=stable[time_base],
            numberOfGaps=gaps[time_base],
        )
        for time_base in granularities
    ]

    return DatetimeColumnTemporalConsistency(periodicity, temporal_consistencies, column)
