import pandas as pd


class NumericColumn:
    def __init__(self, column_name: str, column_data: pd.Series):
        self._column_name = column_name
        self._colum_data = column_data
