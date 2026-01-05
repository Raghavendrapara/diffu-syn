import torch
import torch.optim as optim
from tqdm import tqdm
import polars as pl
import numpy as np

# Import internal physics
from diffusyn.core.model import TabularModel
from diffusyn.core.diffusion import DiffusionEngine
from diffusyn.core.dataset import DiffuSynDataset
from torch.utils.data import DataLoader


class SimpleScaler:
    """
    A lightweight Scaler to map data to [-1, 1] for Neural Networks.
    """

    def __init__(self):
        self.min_vals = None
        self.max_vals = None
        self.columns = None

    def fit(self, df: pl.DataFrame):
        self.columns = df.columns
        # Calculate min/max for every column
        self.min_vals = df.min().to_numpy()[0]
        self.max_vals = df.max().to_numpy()[0]

        self.max_vals = np.where(self.max_vals == self.min_vals, self.min_vals + 1.0, self.max_vals)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formula: 2 * (x - min) / (max - min) - 1
        # Maps [min, max] -> [-1, 1]
        data = df.to_numpy()
        scaled = 2 * (data - self.min_vals) / (self.max_vals - self.min_vals) - 1
        return pl.DataFrame(scaled, schema=self.columns)

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formula: (x + 1) / 2 * (max - min) + min
        data = df.to_numpy()
        unscaled = (data + 1) / 2 * (self.max_vals - self.min_vals) + self.min_vals
        return pl.DataFrame(unscaled, schema=self.columns)

    def to_dict(self):
        return {
            "min_vals": self.min_vals.tolist() if self.min_vals is not None else [],
            "max_vals": self.max_vals.tolist() if self.max_vals is not None else [],
            "columns": self.columns
        }

    def from_dict(self, data):
        self.min_vals = np.array(data["min_vals"])
        self.max_vals = np.array(data["max_vals"])
        self.columns = data["columns"]


class TabularDiffusion:
    """
    The High-Level Interface for Users.
    """

    def __init__(self, device="cpu", lr=1e-3, hidden_dim=128, layers=3, steps=1000):
        self.device = device
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.steps = steps

        self.model = None
        self.diffusion = None
        self.input_dim = None
        self.columns = None
        self.scaler = SimpleScaler()  # <-- NEW: The Normalizer

    def fit(self, data, epochs=5, batch_size=1024):
        """
        Trains the diffusion model on the provided data.
        """
        # 1. Load Data
        if isinstance(data, str):
            df = pl.read_csv(data)
        else:
            df = data

        # 2. Fit Scaler & Normalize Data
        print("Preprocessing: Scaling data to [-1, 1] range...")
        self.scaler.fit(df)
        df_scaled = self.scaler.transform(df)

        dataset = DiffuSynDataset(df_scaled, batch_size=batch_size)
        dataloader = DataLoader(dataset, batch_size=None)

        if self.input_dim is None:
            self.columns = df.columns
            self.input_dim = len(self.columns)
            print(f"Auto-detected Input Dimension: {self.input_dim}")

        # 3. Initialize Engine
        self.model = TabularModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            layers=self.layers
        ).to(self.device)

        self.diffusion = DiffusionEngine(self.model, steps=self.steps, device=self.device)
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

    def generate(self, n_samples: int, batch_size: int = 10000) -> pl.DataFrame:
        """
        Generates synthetic data.
        """
        if not self.model:
            raise RuntimeError("Model is not trained! Call .fit() first.")

        print(f"Generating {n_samples} samples in batches of {batch_size}...")

        generated_batches = []
        remaining = n_samples

        cols = self.columns if self.columns else [f"col_{i}" for i in range(self.input_dim)]

        while remaining > 0:
            current_batch_size = min(remaining, batch_size)

            # Generate raw data (in [-1, 1] range)
            batch_tensor = self.diffusion.sample(current_batch_size, self.input_dim)
            batch_data = batch_tensor.cpu().numpy()

            # Wrap in DF
            batch_df = pl.DataFrame(batch_data, schema=cols)

            batch_unscaled = self.scaler.inverse_transform(batch_df)

            generated_batches.append(batch_unscaled)
            remaining -= current_batch_size

        return pl.concat(generated_batches)

    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler": self.scaler.to_dict(),  # <-- Save scaler stats
            "config": {
                "input_dim": self.input_dim,
                "columns": self.columns,
                "hidden_dim": self.hidden_dim,
                "layers": self.layers,
                "steps": self.steps
            }
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint["config"]["input_dim"]
        self.columns = checkpoint["config"].get("columns")
        self.hidden_dim = checkpoint["config"].get("hidden_dim", self.hidden_dim)
        self.layers = checkpoint["config"].get("layers", self.layers)
        self.steps = checkpoint["config"].get("steps", 1000)

        # Re-init model
        self.model = TabularModel(self.input_dim, self.hidden_dim, self.layers).to(self.device)
        self.diffusion = DiffusionEngine(self.model, steps=self.steps, device=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

        # Load scaler
        if "scaler" in checkpoint:
            self.scaler.from_dict(checkpoint["scaler"])

    def evaluate(self, original_data, synthetic_data, sample_size: int = 50000):
        """
        Evaluates the quality of synthetic data using SDMetrics.
        """
        from sdmetrics.reports.single_table import QualityReport

        def prepare_df(df, limit):
            if isinstance(df, pl.DataFrame):
                if len(df) > limit:
                    df = df.sample(n=limit, with_replacement=False)
                return df.to_pandas()
            return df

        print(f"Sampling data (limit={sample_size}) for evaluation...")
        original_pandas = prepare_df(original_data, sample_size)
        synthetic_pandas = prepare_df(synthetic_data, sample_size)

        metadata = {
            "columns": {col: {"sdtype": "numerical"} for col in original_pandas.columns}
        }

        report = QualityReport()
        report.generate(original_pandas, synthetic_pandas, metadata)

        return {
            "score": report.get_score(),
            "details": report.get_details("Column Shapes")
        }