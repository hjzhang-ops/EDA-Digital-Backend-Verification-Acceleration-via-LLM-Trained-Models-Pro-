
# Model Card

- **Intended Use**: Parse EDA verification logs (timing, power, IR drop, congestion) and classify issues / extract key fields.
- **Training Data**: Synthetic examples provided; replace with proprietary logs for real usage.
- **Risks**: Misclassification on previously unseen report formats; requires validation with golden checks.
- **Metrics**: F1, precision, recall; confusion matrix in `results/metrics/`.
