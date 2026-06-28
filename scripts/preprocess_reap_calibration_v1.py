"""Pre-process 0xSero/reap-calibration-data-v1 into standard REAP format.

Converts the raw JSONL files into a uniform format with `messages` (chat) +
`category` fields, matching the keypa/reaper-calibration schema.

Usage:
    python scripts/preprocess_reap_calibration_v1.py                    # full v1 (includes refusals)
    python scripts/preprocess_reap_calibration_v1.py --use-filtered     # v2 (refusals removed)

Output: artifacts/datasets/reap-calibration-v1/train.jsonl
Ready for upload to HF or local use.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


RAW_REPO = "0xSero/reap-calibration-data-v1"
FILES = {
    "full": "calibration-v1.jsonl",
    "filtered": "filtered_v2.jsonl",
}
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "datasets" / "reap-calibration-v1"
OUTPUT_FILE = "train.jsonl"

DOMAIN_CATEGORIES = [
    "agentic", "coding", "cuda", "cybersecurity", "deep_reasoning",
    "function_calling", "long_context", "math", "science", "terminal",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-filtered", action="store_true",
        help="Use filtered_v2.jsonl (refusals removed) instead of full v1",
    )
    args = parser.parse_args()

    version = "filtered" if args.use_filtered else "full"
    raw_filename = FILES[version]
    label_suffix = " (filtered, no refusals)" if args.use_filtered else " (includes refusals)"

    # Download raw data
    raw_dir = Path("/tmp") / "reap-calibration-data-v1"
    print(f"Downloading {RAW_REPO} to {raw_dir}...")
    snapshot_download(
        repo_id=RAW_REPO,
        repo_type="dataset",
        local_dir=str(raw_dir),
    )

    raw_path = raw_dir / raw_filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Expected {raw_path} not found")

    # Count lines
    total = sum(1 for _ in open(raw_path, encoding="utf-8"))
    print(f"Processing {total} samples from {raw_filename}{label_suffix}")

    # Convert
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_FILE

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

    print(f"Converted {converted} samples, skipped {skipped}")
    print(f"Output: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\nTo upload to HuggingFace:")
    print(f"  huggingface-cli upload keypa/reap-calibration-v1 {out_path} --repo-type dataset")
    print(f"\nTo use locally in the pipeline:")
    print(f"  --dataset-name \"{out_path.parent}\"")


if __name__ == "__main__":
    main()
