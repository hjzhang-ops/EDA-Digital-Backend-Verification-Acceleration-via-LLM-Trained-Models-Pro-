
# EDA Digital Backend Verification Acceleration via LLM-Trained Models (Pro)

This repository provides a **production-grade** template for accelerating **digital backend verification** using **LLM-based analysis**. It includes:
- A **fine-tuning pipeline** for adapting a pretrained transformer to EDA logs;
- A **prediction pipeline** with report parsing, feature extraction, and LLM reasoning;
- A **metric & evaluation module** (precision/recall/F1, confusion matrix);
- A **Streamlit dashboard** for interactive summaries;
- Clear **configs** and **tests** to bootstrap real integrations.

> The code is self-contained with synthetic examples. Replace the stubs with connections to your tools (e.g., Synopsys Fusion Compiler / IC Validator).

## Quickstart
```bash
pip install -r requirements.txt
# 1) Finetune (on synthetic sample)
python src/pipelines/finetune.py --config configs/train.yaml
# 2) Run end-to-end pipeline
python scripts/run_pipeline.py --input data/sample_logs/example_timing.rpt --config configs/infer.yaml
# 3) Launch dashboard
streamlit run src/dashboard_app.py
```
