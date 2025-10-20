
from typing import Dict
import numpy as np

def to_feature_vector(parsed: Dict) -> np.ndarray:
    # Fixed order vector; NaNs -> 0
    keys = ["slack", "congestion_pct", "ir_drop_mv", "power_mw"]
    vec = [float(parsed.get(k, 0.0)) for k in keys]
    return np.array(vec, dtype=float)
