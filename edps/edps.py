import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from edps.date_time_column import DateTimeColumn
from edps.numeric_column import NumericColumn
from edps.schema.pydantic_schema import EDPSchema, StructuredEDPDataSet
from edps.string_column import StringColumn


class _StructuredData:
    def __init__(self, data: pd.DataFrame):
        self._logger = logging.getLogger(__name__)
        self._data = data

    def get_column_types(self) -> Dict:
        columns: Dict = {"text_columns": [], "numeric_columns": [], "datetime_columns": []}

        for column in self._data.columns:
            self._logger.debug(
                "Checking column %s", (column, pd.api.types.infer_dtype(self._data[column], skipna=True))
            )

            kind = self._data[column].dtype.kind
            if kind in "biufc":
                self._logger.info("Assumed type of column '%s' is numeric", column)
                columns["numeric_columns"].append(NumericColumn(column, self._data[column]))

            if kind in "O":
                self._logger.info("Assumed type of column '%s' is string", column)
                try:
                    numeric_column = self._to_numeric(column)
                    columns["numeric_columns"].append(numeric_column)
                    continue
                except:
                    self._logger.warning("Cast to numeric failed for column '%s'", column)

                try:
                    datetime_column = self._to_date_time(column)
                    columns["datetime_columns"].append(datetime_column)
                    continue
                except Exception as e:
                    self._logger.warning("Cast to date/time failed for column '%s': %s" % (column, e))

                self._logger.info("Column '%s' is string", column)
                columns["text_columns"].append(StringColumn(column, self._data[column]))

            if kind in "M":
                try:
                    datetime_column = self._to_date_time(column)
                    columns["datetime_columns"].append(datetime_column)
                    continue
                except Exception as e:
                    self._logger.warning("Cast to date/time failed for column '%s': %s" % (column, e))

                    self._logger.info("Column '%s' is string", column)
                    columns["text_columns"].append(StringColumn(column, self._data[column]))

        return columns

    def _to_date_time(self, column: pd.Series) -> DateTimeColumn:
        dt_elements = pd.to_datetime(self._data[column])
        self._logger.info("Column '%s' is date/time", column)
        return DateTimeColumn(column, dt_elements)

    def _to_numeric(self, column: pd.Series) -> NumericColumn:
        numeric_elements = pd.to_numeric(self._data[column])
        self._logger.info("Column '%s' is numeric", column)
        return NumericColumn(column, numeric_elements)


class EDPS:
    def __init__(self, edp_schema: EDPSchema):
        self._logger = logging.getLogger(__name__)
        self._extensions: List = ["pickle"]
        self._logger.info("Initializing EDPS service. The following extensions are supported: %s", self._extensions)
        self._edp_schema = edp_schema

    def load_files_in_directory(self, directory_path: Path):
        if not directory_path.is_dir():
            error_message = f"Path '{directory_path}' isn't a directory."
            self._logger.error(error_message)
            raise ValueError(error_message)

        self._logger.info("Loading files in directory '%s'", directory_path)

        for extension in self._extensions:
            files = directory_path.glob(f"*.{extension}")
            columns: Dict = {}

            column_list = []
            for file in files:
                self._logger.info("Processing file '%s'", file)
                df = pd.read_pickle(file)
                sd = _StructuredData(df)
                columns = sd.get_column_types()  # TODO: merge

            for dt_col in columns["datetime_columns"]:
                column_list.append(dt_col.get_column_definition())

            self._edp_schema.asset = StructuredEDPDataSet(columnCount=len(column_list), rowCount=50, columns=column_list)

        print(self._edp_schema.model_dump_json())
