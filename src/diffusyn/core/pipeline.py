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
        self.columns = None

    def fit(self, data, epochs=5, batch_size=1024):
        """
        Trains the diffusion model on the provided data.
        data: Can be a file path (str) or a Polars DataFrame.
        """
        dataset = DiffuSynDataset(data, batch_size=batch_size)
        dataloader = DataLoader(dataset, batch_size=None)

        if self.input_dim is None:
            schema = dataset._get_schema_info()
            self.columns = schema.names()
            self.input_dim = len(self.columns)
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

    def generate(self, n_samples: int, batch_size: int = 10000) -> pl.DataFrame:
        """
        Generates synthetic data in batches to avoid OOM errors.
        """
        if not self.model:
            raise RuntimeError("Model is not trained! Call .fit() first.")

        print(f"Generating {n_samples} samples in batches of {batch_size}...")
        
        generated_batches = []
        remaining = n_samples
        
        # Calculate column names once
        cols = self.columns if self.columns else [f"col_{i}" for i in range(self.input_dim)]

        while remaining > 0:
            current_batch_size = min(remaining, batch_size)
            
            # Generate batch on device
            batch_tensor = self.diffusion.sample(current_batch_size, self.input_dim)
            
            # Move to CPU and numpy immediately to free VRAM
            batch_data = batch_tensor.cpu().numpy()
            
            # Wrap in Polars DataFrame
            generated_batches.append(pl.DataFrame(batch_data, schema=cols))
            
            remaining -= current_batch_size

        # Combine all batches
        return pl.concat(generated_batches)

    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "columns": self.columns,
                "hidden_dim": self.hidden_dim,
                "layers": self.layers
            }
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint["config"]["input_dim"]
        self.columns = checkpoint["config"].get("columns")
        self.hidden_dim = checkpoint["config"].get("hidden_dim", self.hidden_dim)
        self.layers = checkpoint["config"].get("layers", self.layers)

        # Re-init model
        self.model = TabularModel(self.input_dim, self.hidden_dim, self.layers).to(self.device)
        self.diffusion = DiffusionEngine(self.model, device=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

    def evaluate(self, original_data, synthetic_data, sample_size: int = 50000):
        """
        Evaluates the quality of synthetic data using SDMetrics.
        Auto-samples data to avoid MemoryErrors with large datasets.
        """
        from sdmetrics.reports.single_table import QualityReport

        # Helper to sample and convert
        def prepare_df(df, limit):
            if isinstance(df, pl.DataFrame):
                if len(df) > limit:
                    df = df.sample(n=limit, with_replacement=False)
                return df.to_pandas()
            # If already pandas or other, assume it fits or user handled it
            return df

        print(f"Sampling data (limit={sample_size}) for evaluation...")
        original_pandas = prepare_df(original_data, sample_size)
        synthetic_pandas = prepare_df(synthetic_data, sample_size)

        # Basic metadata (infer from columns if not provided)
        # For simplicity, we assume all columns are numerical for now
        metadata = {
            "columns": {col: {"sdtype": "numerical"} for col in original_pandas.columns}
        }

        report = QualityReport()
        report.generate(original_pandas, synthetic_pandas, metadata)

        return {
            "score": report.get_score(),
            "details": report.get_details("Column Shapes")
        }