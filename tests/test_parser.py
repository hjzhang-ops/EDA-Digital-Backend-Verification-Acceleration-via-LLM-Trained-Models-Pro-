
from src.verification_parser import parse_verification_report

def test_parse_basic():
    text = "Slack (VIOLATED): -0.200; overflow 9.1% ; IR drop 78mV; power spike to 145mW"
    out = parse_verification_report(text)
    assert round(out['slack'],3) == -0.200
    assert int(out['ir_drop_mv']) == 78
