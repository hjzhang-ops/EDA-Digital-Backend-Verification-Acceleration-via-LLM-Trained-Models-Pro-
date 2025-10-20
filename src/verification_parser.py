
import re
from typing import Dict

def parse_verification_report(text: str) -> Dict:
    # Very simple parser for demo; extend with full grammar for real usage
    result = {}
    slack = re.search(r"Slack \(VIOLATED\):\s*([\-0-9\.]+)", text)
    cong = re.search(r"(?:congestion|overflow)\s*(\d+\.?\d*)%", text, re.I)
    ir = re.search(r"IR drop.*?(\d+)mV", text, re.I)
    power = re.search(r"(?:power spike to|power:)\s*(\d+\.?\d*)mW", text, re.I)

    if slack: result["slack"] = float(slack.group(1))
    if cong: result["congestion_pct"] = float(cong.group(1))
    if ir: result["ir_drop_mv"] = float(ir.group(1))
    if power: result["power_mw"] = float(power.group(1))
    return result
