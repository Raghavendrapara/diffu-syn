# diffu-syn

A PyTorch-based MLP for tabular diffusion that generates synthetic data from private datasets while maintaining privacy guarantees.

## Overview

diffu-syn uses diffusion models to create synthetic tabular data. The generated datasets are audited with SDMetrics to ensure they meet privacy and utility standards.

## Features

- PyTorch-based MLP architecture for tabular data
- Diffusion-based synthetic data generation
- Privacy and utility evaluation via SDMetrics
- Designed for sensitive datasets

## Installation

```bash
git clone https://github.com/raghavendrapara/diffu-syn.git
cd diffu-syn

# Spin up the entire infrastructure (API, Workers, DB)
docker compose up -d --build
```

## Evaluation

Use SDMetrics to assess privacy and utility:

```python
from sdmetrics.evaluation import evaluate

score = evaluate(synthetic_data, original_data)
```

## Requirements

- Python 3.8+
- PyTorch
- SDMetrics

## License

MIT
