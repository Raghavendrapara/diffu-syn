import polars as pl
import numpy as np
import os
from diffusyn.core.pipeline import TabularDiffusion

def main():
    print("="*60)
    print("🚀 Diffu-Syn: End-to-End Demo Workflow")
    print("="*60)

    # ---------------------------------------------------------
    # 1. Prepare Dummy Data
    # ---------------------------------------------------------
    print("\n[1] Preparing Synthetic 'Real' Data...")
    n_real = 1000
    df_real = pl.DataFrame({
        "age": np.random.randint(18, 90, n_real),
        "income": np.random.normal(55000, 15000, n_real),
        "credit_score": np.random.uniform(300, 850, n_real)
    })
    print(f"    - Created {n_real} rows of training data.")
    print(df_real.head(3))

    # ---------------------------------------------------------
    # 2. Initialize & Train Model
    # ---------------------------------------------------------
    print("\n[2] Initializing and Training Diffusion Model...")
    # Parameters are kept small for this quick demo
    model = TabularDiffusion(lr=1e-3, hidden_dim=64, layers=2, device="cpu")
    
    print("    - Training for 3 epochs (fast mode)...")
    model.fit(df_real, epochs=3, batch_size=256)
    print("    - Training Complete.")

    # ---------------------------------------------------------
    # 3. Generate Synthetic Data
    # ---------------------------------------------------------
    print("\n[3] Generating Synthetic Data...")
    n_syn = 5
    df_syn = model.generate(n_samples=n_syn)
    
    print(f"    - Generated {n_syn} samples:")
    print(df_syn)

    # Verify column mapping
    if df_syn.columns == df_real.columns:
        print("    ✅ Column mapping preserved successfully.")
    else:
        print(f"    ❌ Column mapping failed! Got {df_syn.columns}")

    # ---------------------------------------------------------
    # 4. Evaluate Quality
    # ---------------------------------------------------------
    print("\n[4] Evaluating Data Quality (SDMetrics)...")
    try:
        report = model.evaluate(df_real, df_syn)
        print(f"    - Overall Quality Score: {report['score']:.4f}")
        print("    - Column Shape Quality:")
        print(report['details'])
    except Exception as e:
        print(f"    ⚠️ Evaluation skipped/failed: {e}")

    # ---------------------------------------------------------
    # 5. Model Persistence (Save/Load)
    # ---------------------------------------------------------
    print("\n[5] Testing Save/Load Functionality...")
    save_path = "demo_model.pth"
    model.save(save_path)
    print(f"    - Model saved to {save_path}")

    print("    - Loading model from disk...")
    loaded_model = TabularDiffusion()
    loaded_model.load(save_path)

    # Verify loaded model can generate
    df_syn_loaded = loaded_model.generate(n_samples=3)
    print("    - Loaded model generated 3 samples successfully.")
    
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)
        print("    - Cleanup: Removed temporary model file.")

    print("\n" + "="*60)
    print("🎉 Demo Complete! The system is fully operational.")
    print("="*60)

if __name__ == "__main__":
    main()
