import torch
import polars as pl
from torch.utils.data import IterableDataset
import os


class DiffuSynDataset(IterableDataset):
    """
    Universal Dataset that handles both Local Files (Streaming)
    and In-Memory DataFrames (Library usage).
    """

    def __init__(self, data, batch_size: int = 1024, numerical_cols: list = None):
        self.batch_size = batch_size
        self.numerical_cols = numerical_cols

        # 1. Handle File Path (Server/Streaming Mode)
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Dataset not found at {data}")
            self.lazy_df = pl.scan_csv(data)

        # 2. Handle In-Memory DataFrame (Library/Notebook Mode)
        elif isinstance(data, pl.DataFrame):
            self.df_data = data
            self.lazy_df = data.lazy()

        elif isinstance(data, pl.LazyFrame):
            self.lazy_df = data

        else:
            raise ValueError("Data must be a file path (str) or Polars DataFrame")

        # Validate Schema immediately
        self.validate_schema()

    def _get_schema_info(self):
        """Helper to dynamically determine input dimensions."""
        return self.lazy_df.collect_schema()

    def validate_schema(self):
        """
        Ensures all columns are numerical.
        Raises ValueError if categorical/string columns are found.
        """
        schema = self._get_schema_info()
        non_numeric = []
        
        for name, dtype in schema.items():
            # Polars types: Float32, Float64, Int32, Int64, etc.
            if not (dtype.is_numeric() or dtype.is_float() or dtype.is_integer()):
                non_numeric.append(name)
        
        if non_numeric:
            raise ValueError(
                f"Non-numerical columns found: {non_numeric}. "
                "Current version only supports numerical data. "
                "Please preprocess (e.g., OneHotEncode) your data first."
            )

    def preprocess_batch(self, df: pl.DataFrame) -> torch.Tensor:
        if self.numerical_cols:
            df = df.select(self.numerical_cols)
        return torch.tensor(df.to_numpy(), dtype=torch.float32)

    def __iter__(self):
        if hasattr(self, 'path'):
            reader = pl.read_csv_batched(self.path, batch_size=self.batch_size)
            while True:
                batches = reader.next_batches(1)
                if not batches:
                    break
                yield self.preprocess_batch(batches[0])

        elif self.df_data is not None:
            n_rows = len(self.df_data)
            for i in range(0, n_rows, self.batch_size):
                slice_df = self.df_data.slice(i, self.batch_size)
                yield self.preprocess_batch(slice_df)