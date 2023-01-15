import pandas as pd
import pyarrow as pa
from pyarrow import csv
import pyarrow.parquet as pq
from typing import Union
from pathlib import Path


def read_csv(file, sep="\t") -> pd.DataFrame:
    parse_options = csv.ParseOptions(delimiter=sep)
    data = csv.read_csv(file, parse_options=parse_options)
    df = data.to_pandas()
    del data
    return df


def read_table(file: str) -> pd.DataFrame:
    return pq.read_table().to_pandas()


def read(file: Union[str, Path], sep="\t"):
    if isinstance(file, Path):
        file = str(file)
    if file.endswith("csv") or file.endswith("txt"):
        return read_csv(file, sep)
    elif file.endswith("parquet"):
        return read_table(file)
    else:
        raise ValueError("not support format")


def to_csv(df: pd.DataFrame, output: str):
    table = pa.Table.from_pandas(df)
    csv.write_csv(table, output)


def to_parquet(df: pd.DataFrame, output: str):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, "%s.parquet" % output)


def save(df: pd.DataFrame, output: str, file_format: str = "parquet"):
    if file_format == "parquet":
        output = output.replace(".parquet", "")
        to_parquet(df, output)
    elif file_format == "csv" or file_format == "txt":
        output = output.replace(".csv", "")
        output = output.replace(".txt", "")
        to_csv(df, "%s.csv" % output)
    else:
        raise ValueError('not support format')
