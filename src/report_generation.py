
import os, json, csv
from typing import Dict, List

def generate_report(prediction: Dict, structured: Dict, out_dir: str = "results"):
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    jpath = os.path.join(out_dir, "predictions", "latest.json")
    with open(jpath, 'w') as f:
        json.dump({"prediction": prediction, "structured": structured}, f, indent=2)
    # also write minimal csv
    csvp = os.path.join(out_dir, "metrics", "summary.csv")
    with open(csvp, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(["label", "slack", "congestion", "ir_drop", "power"])
        ft = structured.get("features", {})
        w.writerow([prediction.get("label"), ft.get("slack", ""), ft.get("congestion_pct", ""), ft.get("ir_drop_mv", ""), ft.get("power_mw", "")])
    return jpath, csvp
