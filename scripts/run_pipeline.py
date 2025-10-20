
import argparse, yaml
from src.data_preprocess import preprocess_logs
from src.llm_inference import predict_issue
from src.report_generation import generate_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to verification log/report')
    parser.add_argument('--config', default='configs/infer.yaml')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    struct = preprocess_logs(args.input)
    pred = predict_issue(cfg['model_path'], struct['raw_text'], cfg['label_map'], cfg['max_length'], cfg['threshold'])
    jpath, csvp = generate_report(pred, struct, out_dir='results')
    print('[Pipeline] JSON:', jpath)
    print('[Pipeline] CSV:', csvp)

if __name__ == '__main__':
    main()
