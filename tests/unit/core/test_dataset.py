import pytest
import polars as pl
import torch
from diffusyn.core.dataset import DiffuSynDataset

def test_dataset_initialization_dataframe(dummy_data):
    dataset = DiffuSynDataset(dummy_data, batch_size=10)
    schema = dataset._get_schema_info()
    assert len(schema) == 3
    assert schema.names() == ["col_a", "col_b", "col_c"]

def test_dataset_iteration(dummy_data):
    batch_size = 20
    dataset = DiffuSynDataset(dummy_data, batch_size=batch_size)
    
    batches = list(dataset)
    # 100 rows / 20 batch_size = 5 batches
    assert len(batches) == 5
    assert isinstance(batches[0], torch.Tensor)
    assert batches[0].shape == (batch_size, 3)

def test_dataset_file_loading(dummy_data, tmp_path):
    csv_path = tmp_path / "test.csv"
    dummy_data.write_csv(csv_path)
    
    dataset = DiffuSynDataset(str(csv_path), batch_size=10)
    schema = dataset._get_schema_info()
    assert schema.names() == ["col_a", "col_b", "col_c"]
