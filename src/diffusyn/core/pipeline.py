import torch
import torch.optim as optim
from tqdm import tqdm
import polars as pl

# Import internal physics
from diffusyn.core.model import TabularModel
from diffusyn.core.diffusion import DiffusionEngine
from diffusyn.core.dataset import DiffuSynDataset
from torch.utils.data import DataLoader


class TabularDiffusion:
    """
    The High-Level Interface for Users.
    Usage:
        model = TabularDiffusion()
        model.fit(df)
        synthetic_df = model.generate(100)
    """

    def __init__(self, device="cpu", lr=1e-3, hidden_dim=128, layers=3):
        self.device = device
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.model = None
        self.diffusion = None
        self.input_dim = None

    def fit(self, data, epochs=5, batch_size=1024):
        """
        Trains the diffusion model on the provided data.
        data: Can be a file path (str) or a Polars DataFrame.
        """
        dataset = DiffuSynDataset(data, batch_size=batch_size)
        dataloader = DataLoader(dataset, batch_size=None)

        if self.input_dim is None:
            schema = dataset._get_schema_info()
            self.input_dim = len(schema)
            print(f"Auto-detected Input Dimension: {self.input_dim}")

        # 3. Initialize Engine
        self.model = TabularModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            layers=self.layers
        ).to(self.device)

        self.diffusion = DiffusionEngine(self.model, device=self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 4. Training Loop
        self.model.train()
        print(f"Starting Training for {epochs} epochs...")

        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = self.diffusion.compute_loss(batch)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    def generate(self, n_samples: int) -> pl.DataFrame:
        """
        Generates synthetic data.
        """
        if not self.model:
            raise RuntimeError("Model is not trained! Call .fit() first.")

        print(f"Generating {n_samples} samples...")
        synthetic_tensor = self.diffusion.sample(n_samples, self.input_dim)

        # Convert back to Polars
        data = synthetic_tensor.cpu().numpy()
        # TODO: Ideally we map these back to original column names
        cols = [f"col_{i}" for i in range(self.input_dim)]
        return pl.DataFrame(data, schema=cols)

    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {"input_dim": self.input_dim}
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint["config"]["input_dim"]

        # Re-init model
        self.model = TabularModel(self.input_dim, self.hidden_dim, self.layers).to(self.device)
        self.diffusion = DiffusionEngine(self.model, device=self.device)
        self.model.load_state_dict(checkpoint["model_state"])