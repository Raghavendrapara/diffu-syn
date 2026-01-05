import pytest
import polars as pl
import numpy as np

@pytest.fixture
def dummy_data():
    """Creates a small dummy dataset for testing."""
    return pl.DataFrame({
        "col_a": np.random.randn(100),
        "col_b": np.random.randint(0, 10, 100),
        "col_c": np.random.uniform(0, 1, 100)
    })

@pytest.fixture
def anyio_backend():
    return 'asyncio'
