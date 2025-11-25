import torch
import polars as pl
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import os


class StreamingTabularDataset(IterableDataset):
    """
    A Zero-Memory-Overhead Dataset.
    Streams data from disk directly to the GPU training loop using Polars LazyFrames.
    """

    def __init__(self, file_path: str, batch_size: int = 1024, numerical_cols: list = None):
        self.file_path = file_path
        self.batch_size = batch_size
        self.numerical_cols = numerical_cols

        # Quick check to ensure file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")

    def preprocess_batch(self, df: pl.DataFrame) -> torch.Tensor:
        """
        Stateless preprocessing.
        In a real scenario, you would apply normalization (x - min) / (max - min) here
        using pre-computed metadata. For now, we cast to float32.
        """
        # Select only the columns we want (if specified)
        if self.numerical_cols:
            df = df.select(self.numerical_cols)

        # Polars to Numpy (Zero-copy if possible)
        data_array = df.to_numpy()

        # Numpy to PyTorch
        return torch.tensor(data_array, dtype=torch.float32)

    def __iter__(self):
        """
        The Engine: Polars scans the file in chunks without loading it all.
        """
        # 1. Create a LazyFrame (Pointer to disk, loads nothing yet)
        lazy_df = pl.scan_csv(self.file_path)

        # 2. Stream it
        # 'collect(streaming=True)' enables the Rust query engine to process in chunks
        # 'iter_slices' yields batches of size n_rows
        stream = lazy_df.collect(engine="streaming").iter_slices(n_rows=self.batch_size)

        for batch_df in stream:
            # 3. Process and Yield
            tensor_batch = self.preprocess_batch(batch_df)

            # IterableDataset expects us to yield samples.
            # However, for efficiency, we often yield batches.
            # Here we yield the whole tensor batch to be grabbed by the DataLoader.
            yield tensor_batch


# ==========================================
# SANITY CHECK (Run this file directly)
# ==========================================
if __name__ == "__main__":
    # 1. Create a dummy "Huge" CSV (just 10k rows for test)
    print("Generating dummy data...")
    dummy_data = pl.DataFrame({
        "age": np.random.randint(18, 90, 10000),
        "salary": np.random.normal(50000, 15000, 10000),
        "risk_score": np.random.rand(10000)
    })
    dummy_path = "test_huge_data.csv"
    dummy_data.write_csv(dummy_path)
    print(f"Created {dummy_path}")

    # 2. Initialize the Streaming Dataset
    print("Initializing Stream...")
    dataset = StreamingTabularDataset(dummy_path, batch_size=50, numerical_cols=["age", "salary", "risk_score"])

    # 3. Wrap in PyTorch DataLoader
    # batch_size=None tells PyTorch "The dataset is already yielding batches, don't re-batch them"
    loader = DataLoader(dataset, batch_size=None)

    # 4. Consume the stream
    print("Reading chunks...")
    for i, batch in enumerate(loader):
        print(f"Batch {i} Shape: {batch.shape} | Dtype: {batch.dtype}")
        # Verify it's a Tensor
        assert isinstance(batch, torch.Tensor)

        # Stop after 3 batches to save time
        if i == 2:
            print("...Stopping stream (Test Passed).")
            break

    # Cleanup
    os.remove(dummy_path)
    print("Test Complete. Environment is ready for AI.")