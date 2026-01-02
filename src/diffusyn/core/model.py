import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes the integer time step (e.g., t=500) into a dense vector.
    This allows the MLP to understand 'Time' as a distinct feature.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    A building block that allows gradients to flow through the network easily.
    Logic: Output = Input + Layer(Input)
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return self.norm(out + residual)


class TabularModel(nn.Module):
    """
    The Main Brain.
    Takes (Noisy Data + Time) -> Outputs (Predicted Noise)
    """

    def __init__(self, input_dim, hidden_dim=128, layers=3, dropout=0.1):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.network = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):

        t_emb = self.time_mlp(t)
        x_emb = self.input_proj(x)

        h = x_emb + t_emb

        for layer in self.network:
            h = layer(h)

        return self.output_proj(h)


if __name__ == "__main__":
    print("Initializing Tabular Model...")

    # Define dimensions
    batch_size = 32
    input_features = 10

    model = TabularModel(input_dim=input_features, hidden_dim=64, layers=2)
    print("Model Architecture Created.")

    x = torch.randn(batch_size, input_features)  # Noisy Data
    t = torch.randint(0, 1000, (batch_size,))  # Random Time Steps

    output = model(x, t)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")

    if output.shape == x.shape:
        print("Test Complete. Brain is ready.")
    else:
        print(f"CRITICAL FAIL. Output shape {output.shape} does not match Input {x.shape}")