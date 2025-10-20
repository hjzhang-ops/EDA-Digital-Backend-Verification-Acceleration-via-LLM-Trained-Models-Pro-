
from typing import List, Dict
from .verification_parser import parse_verification_report

def preprocess_logs(path: str) -> Dict:
    with open(path, 'r') as f:
        text = f.read()
    fields = parse_verification_report(text)
    return {"raw_text": text, "features": fields}
