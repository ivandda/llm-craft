"""Submit src/sft/train.py as a Vertex AI Custom Training Job.

The training container reads data and writes outputs through the Cloud Storage
FUSE mount that Vertex exposes at /gcs/<bucket>/, so train.py runs unchanged:
inputs come from gs://<bucket>/datasets/... and the run dir lands under
gs://<bucket>/runs/.

train.py is config-driven: it loads a YAML (--config) and accepts per-field
overrides with underscored flags (--train_path, --model_name_or_path, ...).

Run locally (the 'vertex' group provides google-cloud-aiplatform on demand):
    uv run --group vertex python -m src.sft.vertex_submit --run-name qwen05b-10k
"""

import argparse
from datetime import datetime, timezone

from google.cloud import aiplatform

PROJECT = "nlp2026-498021"
REGION = "us-central1"
BUCKET = "llm-craft-bucket"
IMAGE_URI = (
    f"{REGION}-docker.pkg.dev/{PROJECT}/llm-craft-registry/llm-craft-sft:latest"
)
GCS_ROOT = f"/gcs/{BUCKET}"


def default_run_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M_sft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None, help="run_name label for the output dir.")
    parser.add_argument("--image-uri", default=IMAGE_URI)
    parser.add_argument("--machine-type", default="g2-standard-8")
    parser.add_argument("--accelerator-type", default="NVIDIA_L4")
    parser.add_argument(
        "--accelerator-count",
        type=int,
        default=1,
        help="Set 0 for a CPU-only machine (e.g. smoke tests without GPU quota).",
    )

    # train.py passthrough. Paths default to the GCS FUSE mount.
    parser.add_argument(
        "--config",
        default="configs/sft/default.yaml",
        help="YAML config baked into the image that train.py loads.",
    )
    parser.add_argument("--model-name", default=None, help="Override model_name_or_path.")
    parser.add_argument(
        "--train-path", default=f"{GCS_ROOT}/datasets/final-10k/train.jsonl"
    )
    parser.add_argument(
        "--dev-path", default=f"{GCS_ROOT}/datasets/final-10k/dev.jsonl"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps (>0 caps steps; -1 trains by epochs).",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded verbatim to train.py after a '--' separator.",
    )
    return parser.parse_args()


def build_train_args(args: argparse.Namespace, run_name: str) -> list[str]:
    train_args = [
        "--config", args.config,
        "--train_path", args.train_path,
        "--dev_path", args.dev_path,
        "--output_dir", f"{GCS_ROOT}/runs",
        "--run_name", run_name,
    ]
    if args.model_name is not None:
        train_args += ["--model_name_or_path", args.model_name]
    if args.max_steps is not None:
        train_args += ["--max_steps", str(args.max_steps)]
    # argparse.REMAINDER keeps a leading "--"; drop it before forwarding.
    extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
    return train_args + extra


def main() -> None:
    args = parse_args()
    run_name = args.run_name or default_run_name()

    aiplatform.init(
        project=PROJECT, location=REGION, staging_bucket=f"gs://{BUCKET}"
    )

    machine_spec: dict = {"machine_type": args.machine_type}
    # Omit accelerator fields for CPU-only runs (e.g. smoke tests without GPU quota).
    if args.accelerator_count > 0:
        machine_spec["accelerator_type"] = args.accelerator_type
        machine_spec["accelerator_count"] = args.accelerator_count

    worker_pool_specs = [
        {
            "machine_spec": machine_spec,
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.image_uri,
                "command": ["python", "-m", "src.sft.train"],
                "args": build_train_args(args, run_name),
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=run_name, worker_pool_specs=worker_pool_specs
    )
    print(f"Submitting Vertex CustomJob '{run_name}'")
    print(f"  image:  {args.image_uri}")
    print(f"  output: gs://{BUCKET}/runs/ (run dir created by train.py)")
    job.run()


if __name__ == "__main__":
    main()
