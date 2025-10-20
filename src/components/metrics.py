
import json, os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_predictions(labels, preds, out_dir='results/metrics'):
    os.makedirs(out_dir, exist_ok=True)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    with open(os.path.join(out_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    cm = confusion_matrix(labels, preds).tolist()
    with open(os.path.join(out_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump({"confusion_matrix": cm}, f, indent=2)
    return report, cm
