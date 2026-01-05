import pytest
import polars as pl
import numpy as np
import os
import torch
from diffusyn.core.pipeline import TabularDiffusion

def test_pipeline_initialization():
    model = TabularDiffusion(hidden_dim=32, layers=1)
    assert model.model is None
    assert model.input_dim is None

def test_training_and_generation(dummy_data):
    model = TabularDiffusion(hidden_dim=32, layers=1, device="cpu")
    
    # Test Fit
    model.fit(dummy_data, epochs=1, batch_size=50)
    assert model.input_dim == 3
    assert model.columns == ["col_a", "col_b", "col_c"]
    assert model.model is not None

    # Test Generate
    n_samples = 10
    synthetic = model.generate(n_samples)
    
    assert isinstance(synthetic, pl.DataFrame)
    assert synthetic.shape == (n_samples, 3)
    assert synthetic.columns == ["col_a", "col_b", "col_c"]

def test_save_load(dummy_data, tmp_path):
    # Train
    model = TabularDiffusion(hidden_dim=32, layers=1, device="cpu")
    model.fit(dummy_data, epochs=1)
    
    # Save
    save_path = tmp_path / "test_model.pth"
    model.save(save_path)
    assert os.path.exists(save_path)

    # Load
    new_model = TabularDiffusion(device="cpu")
    new_model.load(save_path)
    
    assert new_model.input_dim == 3
    assert new_model.columns == ["col_a", "col_b", "col_c"]
    
    # Verify Loaded Generation
    synthetic = new_model.generate(5)
    assert synthetic.shape == (5, 3)

def test_evaluation(dummy_data):
    model = TabularDiffusion(hidden_dim=32, layers=1, device="cpu")
    model.fit(dummy_data, epochs=1)
    synthetic = model.generate(len(dummy_data))
    
    report = model.evaluate(dummy_data, synthetic)
    assert "score" in report
    assert 0.0 <= report["score"] <= 1.0
    assert "details" in report
