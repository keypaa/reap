"""Pre-process 0xSero/reap-calibration-data-v1 into standard REAP format.

Converts the raw JSONL files into two datasets with `messages` + `category`
fields, matching the keypa/reaper-calibration schema.

Outputs:
    artifacts/datasets/reap-calibration-v1-full/train.jsonl     (23,088, includes refusals)
    artifacts/datasets/reap-calibration-v1-filtered/train.jsonl (20,980, no refusals)

Upload each to HF, then use via composite spec:
    --dataset-name "keypa/reap-calibration-v1-full:100,keypa/reap-calibration-v1-filtered:200"
"""

import json
from pathlib import Path

from huggingface_hub import snapshot_download


RAW_REPO = "0xSero/reap-calibration-data-v1"
FILES = {
    "reap-calibration-v1-full": "calibration-v1.jsonl",
    "reap-calibration-v1-filtered": "filtered_v2.jsonl",
}
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "datasets"

DOMAIN_CATEGORIES = [
    "agentic", "coding", "cuda", "cybersecurity", "deep_reasoning",
    "function_calling", "long_context", "math", "science", "terminal",
]


def process(raw_dir: Path, dataset_name: str, raw_filename: str):
    raw_path = raw_dir / raw_filename
    total = sum(1 for _ in open(raw_path, encoding="utf-8"))
    print(f"\nProcessing {total} samples from {raw_filename} → {dataset_name}")

    out_dir = ARTIFACTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    converted = 0
    skipped = 0
    with open(raw_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            sample = json.loads(line)
            domain = sample.get("domain")
            text = sample.get("text", "")

            if domain not in DOMAIN_CATEGORIES:
                skipped += 1
                continue

            record = {
                "messages": [
                    {"role": "user", "content": text},
                ],
                "category": domain,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted += 1

    print(f"  Converted {converted}, skipped {skipped}")
    print(f"  Output: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return converted


def main():
    raw_dir = Path("/tmp") / "reap-calibration-data-v1"
    print(f"Downloading {RAW_REPO} to {raw_dir}...")
    snapshot_download(
        repo_id=RAW_REPO,
        repo_type="dataset",
        local_dir=str(raw_dir),
    )

    total = 0
    for dataset_name, raw_filename in FILES.items():
        total += process(raw_dir, dataset_name, raw_filename)

    print(f"\nTotal: {total} samples across {len(FILES)} datasets")
    print(f"\nTo upload both to HuggingFace:")
    for dataset_name in FILES:
        out_path = ARTIFACTS_DIR / dataset_name / "train.jsonl"
        print(f"  huggingface-cli upload keypa/{dataset_name} {out_path} --repo-type dataset")

    print(f"\nThen use in the pipeline:")
    print(f'  --dataset-name "keypa/reap-calibration-v1-full:100,keypa/reap-calibration-v1-filtered:200"')


if __name__ == "__main__":
    main()
