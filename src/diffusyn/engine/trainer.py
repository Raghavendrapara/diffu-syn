import torch
import torch.optim as optim
from tqdm import tqdm
import os

from diffusyn.engine.model import TabularModel
from diffusyn.engine.diffusion import DiffusionEngine
from diffusyn.engine.data_loader import StreamingTabularDataset
from torch.utils.data import DataLoader


def train_model(
        data_path: str,
        output_path: str = "model_weights.pth",
        input_dim: int = 10,  # <--- This argument was being ignored before
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu"
):
    print(f"🚀 Starting Training on device: {device}")

    # 1. Setup the Brain
    # FIXED: Changed 'input_features' to 'input_dim'
    model = TabularModel(input_dim=input_dim, hidden_dim=128, layers=3).to(device)

    # 2. Setup the Physics
    diffusion = DiffusionEngine(model, steps=1000, device=device)

    # 3. Setup the Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Setup Data Stream
    dataset = StreamingTabularDataset(data_path, batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None)

    # 5. THE LOOP
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0

        # We wrap the dataloader but handle empty file cases gracefully
        try:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = diffusion.compute_loss(batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({"Loss": loss.item()})
        except Exception as e:
            print(f"⚠️ Batch Error: {e}")
            continue

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"✅ Epoch {epoch + 1} Complete. Average Loss: {avg_loss:.6f}")

    # 6. Save the Brain
    torch.save(model.state_dict(), output_path)
    print(f"💾 Model saved to {output_path}")


# Dummy execution block for local testing
if __name__ == "__main__":
    import polars as pl
    import numpy as np

    dummy_path = "train_data.csv"
    input_dim = 5  # Define it here for the script

    df = pl.DataFrame(np.random.randn(100, input_dim), schema=[f"col_{i}" for i in range(input_dim)])
    df.write_csv(dummy_path)

    train_model(dummy_path, input_dim=input_dim, epochs=1)

    if os.path.exists(dummy_path):
        os.remove(dummy_path)