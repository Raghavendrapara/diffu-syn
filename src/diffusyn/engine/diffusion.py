import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


def get_beta_schedule(steps=1000, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, steps)


class SinusoidalPositionEmbeddings(nn.Module):
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


class DiffusionEngine(nn.Module):
    def __init__(self, model, steps=1000, device="cpu"):
        super().__init__()
        self.model = model
        self.steps = steps
        self.device = device

        # Define the Noise Schedule (The Physics)
        self.betas = get_beta_schedule(steps=steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # --- Pre-calculate Math for the Reverse Process (Generation) ---
        # Equation: x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_bar) * epsilon)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (
                    1.0 - torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])) / (
                                              1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """ Forward Process (Add Noise) """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t])[:, None]
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def compute_loss(self, x_0):
        """ Training Loss """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.steps, (batch_size,), device=self.device)
        noise = torch.randn_like(x_0)
        x_t, real_noise = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, real_noise)

    @torch.no_grad()
    def sample(self, n_samples, input_dim):
        """
        The Reverse Process (Generation).
        Starts with pure Gaussian noise and iteratively 'denoises' it.
        """
        self.model.eval()
        # 1. Start with pure noise (The "Static")
        x = torch.randn((n_samples, input_dim)).to(self.device)

        # 2. Loop backwards from T=1000 down to 0
        for i in tqdm(reversed(range(0, self.steps)), desc="Generating Data", total=self.steps):
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)

            # Predict the noise using the trained Brain
            predicted_noise = self.model(x, t)

            # Get the math constants for this specific time step
            betas_t = self.betas[t][:, None]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
            sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None]

            # The Magic Equation (Langevin Dynamics)
            # Remove a little bit of noise...
            model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

            if i > 0:
                # Add a tiny bit of randomness back (Langevin noise) to keep it looking natural
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(self.posterior_variance[t][:, None])
                x = model_mean + sigma_t * noise
            else:
                x = model_mean

        return x