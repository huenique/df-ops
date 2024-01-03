from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd

import df_ops


@dataclass
class Csv:
    data: str


@dataclass
class Json:
    data: str


type Format = Csv | Json


def create_format_from_string(data_str: str) -> Format:
    """Create a Format (Csv or Json) from a string.

    Args:
        data_str (str): The data in string format.

    Returns:
        Format: The data in either CSV or JSON format.

    Raises:
        ValueError: If the input data is empty or in an invalid format.
    """
    data_str = data_str.strip()
    if not data_str:
        raise ValueError("Input data cannot be empty.")

    try:
        if data_str.startswith("{") and data_str.endswith("}"):
            json.loads(data_str)
            return Json(data_str)
        else:
            csv.Sniffer().sniff(data_str)
            return Csv(data_str)
    except (json.JSONDecodeError, csv.Error):
        raise ValueError("Invalid data format.")


def create_dataframe_from_format(data_str: str) -> pd.DataFrame:
    """Create a Pandas DataFrame from a given format.

    This function takes a string representation of data in either CSV or JSON format,
    and converts it into a Pandas DataFrame. It uses the `pd.read_csv` function for CSV data
    and the `pd.read_json` function for JSON data.

    Args:
        data_str (str): The data in either CSV or JSON format.

    Returns:
        pd.DataFrame: DataFrame created from the provided data.
    """
    data_format = create_format_from_string(data_str)
    match data_format:
        case Csv(data):
            df = pd.read_csv(StringIO(data))  # type: ignore
        case Json(data):
            df = pd.read_json(StringIO(data))  # type: ignore

    return df


class DataFrameOperations(df_ops.DataFrameOperations):
    def add_log_column(self, df: str, column_name: str) -> str:
        """
        Add a new column to the DataFrame with the log of the specified column.

        Parameters:
            df (str): The data in string format (CSV or JSON).
            column_name (str): The name of the column to take the log of.

        Returns:
            str: The DataFrame with the new log column added in JSON format.

        Raises:
            None

        Examples:
            >>> df = 'A,B\\n1,4\\n2,5\\n3,6'
            >>> add_log_column(df, 'A')
            # Output:
            #    A  B  Log_A
            # 0  1  4    0.0
            # 1  2  5    0.693147
            # 2  3  6    1.098612
        """
        df_ = create_dataframe_from_format(df)

        if column_name in df_.columns:
            df_[f"Log_{column_name}"] = np.log(np.array(df_.get(column_name)))
        else:
            print(f"Column '{column_name}' not found in DataFrame.")

        return df_.to_json()  # type: ignore

    def calculate_means(self, df: str) -> str:
        """
        Calculate and return the mean of each column in the DataFrame.

        Parameters:
            df (str): The input DataFrame in string format.

        Returns:
            str: A JSON string containing the mean of each column.
        """
        df_ = create_dataframe_from_format(df)

        return df_.mean().to_json()  # type: ignore

    def multiply_dataframe(self, df: str, factor: float) -> str:
        """Multiply all elements in the DataFrame by a specified factor.

        Args:
            df (str): The DataFrame to be multiplied.
            factor (float): The factor to multiply the DataFrame by.

        Returns:
            str: The resulting DataFrame after multiplication in JSON format.
        """
        df_ = create_dataframe_from_format(df)

        return (df_ * factor).to_json()  # type: ignore

    def read_dataframe(self, path: str = "1,2,3;4,5,6") -> str:
        """
        Read a DataFrame from user input.

        Args:
            path (str): Input data in the format of "1,2,3;4,5,6".

        Returns:
            str: JSON representation of the DataFrame created from the input data.
        """
        return pd.DataFrame([list(map(int, row.split(","))) for row in path.split(";")]).to_json()  # type: ignore
