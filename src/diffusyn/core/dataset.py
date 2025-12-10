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
            self.lazy_df = data.lazy()

        elif isinstance(data, pl.LazyFrame):
            self.lazy_df = data

        else:
            raise ValueError("Data must be a file path (str) or Polars DataFrame")

    def _get_schema_info(self):
        """Helper to dynamically determine input dimensions."""
        # Peek at the schema to get column count
        # Note: In Polars Lazy, we might need to fetch schema from plan
        return self.lazy_df.collect_schema()

    def preprocess_batch(self, df: pl.DataFrame) -> torch.Tensor:
        if self.numerical_cols:
            df = df.select(self.numerical_cols)
        return torch.tensor(df.to_numpy(), dtype=torch.float32)

    def __iter__(self):
        stream = self.lazy_df.collect(engine="streaming").iter_slices(n_rows=self.batch_size)
        for batch_df in stream:
            yield self.preprocess_batch(batch_df)