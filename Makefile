
.PHONY: train infer dash

train:
	python src/pipelines/finetune.py --config configs/train.yaml

infer:
	python scripts/run_pipeline.py --input data/sample_logs/example_timing.rpt --config configs/infer.yaml

dash:
	streamlit run src/dashboard_app.py
