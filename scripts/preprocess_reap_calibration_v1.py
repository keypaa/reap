"""Pre-process 0xSero/reap-calibration-data-v1 into standard REAP format.

Converts the raw JSONL files (with domain-specific nested JSON in the `text`
field) into a uniform format with `messages` (chat) + `category` fields,
matching the keypa/reaper-calibration schema.

Output: two JSONL files ready for load_dataset("json", data_files=...) or
upload to HuggingFace.
"""

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


RAW_REPO = "0xSero/reap-calibration-data-v1"
RAW_FILE = "filtered_v2.jsonl"  # filtered version: no refusals
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "datasets" / "reap-calibration-v1"
OUTPUT_FILE = "train.jsonl"

DOMAIN_CATEGORIES = [
    "agentic", "coding", "cuda", "cybersecurity", "deep_reasoning",
    "function_calling", "long_context", "math", "science", "terminal",
]


def main():
    # Download raw data to temp
    raw_dir = Path("/tmp") / "reap-calibration-data-v1"
    print(f"Downloading {RAW_REPO} to {raw_dir}...")
    snapshot_download(
        repo_id=RAW_REPO,
        repo_type="dataset",
        local_dir=str(raw_dir),
    )

    raw_path = raw_dir / RAW_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"Expected {raw_path} not found")

    # Count lines first
    total = sum(1 for _ in open(raw_path, encoding="utf-8"))
    print(f"Processing {total} samples...")

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

            # Use the raw text as a user message. The text may be
            # JSON-stringified or plain text -- either is fine for
            # calibration (the observer only needs activation data).
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
    print(f"  huggingface-cli upload <your-username>/reap-calibration-v1 {out_path} --repo-type dataset")
    print(f"\nTo use locally in the pipeline:")
    print(f"  --dataset-name \"{out_path.parent}\"")


if __name__ == "__main__":
    main()
