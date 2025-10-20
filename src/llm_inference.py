
from typing import Dict
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path)
    mdl.eval()
    return tok, mdl

def predict_issue(model_path: str, text: str, label_map_path: str, max_length: int = 512, threshold: float = 0.5) -> Dict:
    tok, mdl = load_model(model_path)
    enc = tok(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(**enc).logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
    labels = json.load(open(label_map_path))["classes"]
    idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {"label": labels[idx], "probs": probs, "labels": labels}
