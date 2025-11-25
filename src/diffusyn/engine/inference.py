import torch
import polars as pl
import os

from diffusyn.engine.model import TabularModel
from diffusyn.engine.diffusion import DiffusionEngine


def generate_synthetic_data(
        model_path="model_weights.pth",
        output_csv="synthetic_output.csv",
        n_samples=100,
        input_dim=5,  # MUST match the training data dimension
        device="cpu"
):
    print(f"🔮 Loading Brain from {model_path}...")

    # 1. Rebuild the Architecture (Must match training)
    model = TabularModel(input_dim=input_dim, hidden_dim=128, layers=3).to(device)

    # 2. Load the Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found! Did you run training?")

    model.load_state_dict(torch.load(model_path, map_location=device))

    # 3. Initialize Physics Engine
    diffusion = DiffusionEngine(model, steps=1000, device=device)

    # 4. Generate!
    print(f"⚡ Generating {n_samples} synthetic rows...")
    synthetic_tensor = diffusion.sample(n_samples, input_dim)

    # 5. Save to CSV
    print("💾 Saving to disk...")
    synthetic_data = synthetic_tensor.cpu().numpy()

    # Create column names
    cols = [f"col_{i}" for i in range(input_dim)]
    df = pl.DataFrame(synthetic_data, schema=cols)
    df.write_csv(output_csv)

    print(f"✅ Done! Synthetic data saved to {output_csv}")
    print(df.head())


if __name__ == "__main__":
    # Run generation
    generate_synthetic_data(n_samples=20)