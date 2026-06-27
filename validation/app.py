import json
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_parts(
        packages=[
            "torch>=2.5.0",
            "transformers>=4.57.1",
            "accelerate>=1.3.0",
            "safetensors>=0.5.0",
        ],
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

app = modal.App("v4-weight-validation")


@app.cls(
    image=image,
    gpu="A100-80GB:1",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-token", required=False)],
)
class V4WeightValidator:
    @modal.enter()
    def load_validator(self):
        from validate_fp4 import run_all
        self._run_all = run_all

    @modal.method()
    def validate(self, model_name: str = "deepseek-ai/DeepSeek-V4-Flash"):
        results = self._run_all(model_name)
        path = Path("/tmp/validation-results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {path}")
        return results


@app.local_entrypoint()
def main(model_name: str = "deepseek-ai/DeepSeek-V4-Flash"):
    validator = V4WeightValidator()
    results = validator.validate.remote(model_name)
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2, default=str))
