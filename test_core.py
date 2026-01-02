import polars as pl
import numpy as np
from diffusyn.core import TabularDiffusion

# 1. Create Dummy Data
df = pl.DataFrame({
    "age": np.random.randint(18, 90, 1000),
    "salary": np.random.normal(50000, 15000, 1000),
    "score": np.random.rand(1000)
})
print(df.head(10))
print("Data Created")

# 2. Initialize Model
model = TabularDiffusion()

# 3. Train
print("Training...")
model.fit(df, epochs=1)

# 4. Generate
print("Generating...")
synthetic = model.generate(n_samples=5)
print(synthetic)

print("Success! Core library is decoupled.")