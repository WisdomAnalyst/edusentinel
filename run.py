"""
EduSentinel — One-command bootstrap
Generates data → trains models → embeds curriculum → launches dashboard + API
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
os.chdir(ROOT)


def run(cmd: str, description: str):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"[WARN] Step failed (exit {result.returncode}) — continuing")


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          EduSentinel  ·  Kotlead                             ║
║  AI Platform for Education Access Intelligence               ║
║  Protecting Every Child's Right to Learn in Nigeria          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Step 1 — generate data
    run(
        f"{sys.executable} -m data.synthetic.generate_nigeria_data",
        "Step 1/4: Generating synthetic Nigeria education dataset"
    )

    # Step 2 — train ML models
    run(
        f"{sys.executable} -m models.dropout_prediction.mlflow_pipeline",
        "Step 2/4: Training dropout prediction model (XGBoost + LightGBM + SHAP)"
    )
    run(
        f"{sys.executable} -c \"from data.processing.preprocessor import load_lga_master; from models.risk_intelligence.hotspot_model import run_full_pipeline; import pandas as pd; lga = load_lga_master(); run_full_pipeline(lga); print('Risk intelligence model trained.')\"",
        "Step 3/4: Training risk intelligence & hotspot clustering model"
    )

    # Step 4 — launch dashboard
    print(f"\n{'='*60}")
    print("  Step 4/4: Launching EduSentinel Dashboard")
    print(f"{'='*60}")
    print("\n  Dashboard → http://localhost:8501")
    print("  API docs  → http://localhost:8000/docs")
    print("  MLflow    → http://localhost:5000\n")

    # Launch API in background, dashboard in foreground
    api_proc = subprocess.Popen(
        f"{sys.executable} -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload",
        shell=True, cwd=ROOT,
    )
    try:
        subprocess.run(
            f"streamlit run dashboard/app.py --server.port 8501",
            shell=True, cwd=ROOT,
        )
    finally:
        api_proc.terminate()


if __name__ == "__main__":
    main()
