import logging
from typing import List

import numpy as np
import pandas as pd

from edps.schema.pydantic_schema import (
    DateTimeColumnDefinition,
    TemporalConsistency,
    TemporalConsistencyFrequency,
)


class DateTimeColumn:
    def __init__(self, column_name: str, column_data: pd.Series):
        self._logger = logging.getLogger(__name__)
        self._column_name = column_name

        self._column_definition: DateTimeColumnDefinition = None
        self._temporal_consistency: List[TemporalConsistency] = []

        self._calculate_time_density(column_data)

        self._column_definition = DateTimeColumnDefinition(
            granularity=1, temporalConsistency=self._temporal_consistency
        )

    def get_column_definition(self) -> DateTimeColumnDefinition:
        return self._column_definition

    def _calculate_time_density(self, column_data: pd.Series):
        column_data.index = pd.DatetimeIndex(column_data)
        for density in TemporalConsistencyFrequency:
            values = np.unique(list(column_data.resample(density.value).count()))

            self._logger.debug(
                "Found %d unique values in column '%s' (density '%s')" % (len(values), self._column_name, density.value)
            )

            differrentAbundancies = len(values)
            self._temporal_consistency.append(
                TemporalConsistency(
                    name=None,
                    frequency=density,
                    stable=(differrentAbundancies == 1),
                    differentAbundancies=differrentAbundancies,
                )
            )
