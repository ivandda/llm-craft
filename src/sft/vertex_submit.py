"""Submit a Python module as a Vertex AI Custom Training Job.

The training container reads data and writes outputs through the Cloud Storage
FUSE mount that Vertex exposes at /gcs/<bucket>/. By default this launches
src.sft.train, but other repo modules such as src.sft.eval_base and
src.eval.run_sft_eval can be run too.

train.py is config-driven: it loads a YAML (--config) and accepts per-field
overrides with underscored flags (--train_path, --model_name_or_path, ...).

Run locally (the 'vertex' group provides google-cloud-aiplatform on demand):
    uv run --group vertex python -m src.sft.vertex_submit --run-name qwen05b-10k
"""

import argparse
import contextlib
from datetime import datetime, timezone

from google.cloud import aiplatform

PROJECT = "nlp2026-498021"
REGION = "us-central1"
BUCKET = "llm-craft-bucket"


def default_image_uri(region: str) -> str:
    return f"{region}-docker.pkg.dev/{PROJECT}/llm-craft-registry/llm-craft-sft:latest"


def default_run_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M_sft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None, help="run_name label for the output dir.")
    parser.add_argument("--region", default=REGION, help="Vertex AI region for the CustomJob.")
    parser.add_argument("--bucket", default=BUCKET, help="Cloud Storage bucket mounted by Vertex at /gcs/<bucket>.")
    parser.add_argument("--image-uri", default=None, help="Override the container image URI. Defaults to the registry in --region.")
    parser.add_argument(
        "--module",
        default="src.sft.train",
        help="Python module to run inside the Vertex container.",
    )
    parser.add_argument("--machine-type", default="g2-standard-8")
    parser.add_argument("--accelerator-type", default="NVIDIA_L4")
    parser.add_argument(
        "--accelerator-count",
        type=int,
        default=1,
        help="Set 0 for a CPU-only machine (e.g. smoke tests without GPU quota).",
    )
    parser.add_argument(
        "--boot-disk-gb",
        type=int,
        default=200,
        help="Worker boot disk size in GB. The Vertex default (100) can run low when the base model download plus staged run_dir share the disk.",
    )
    parser.add_argument("--boot-disk-type", default="pd-ssd")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit the job and return immediately instead of streaming until it finishes. "
        "Use for fire-and-forget batches so the local process is not a single point of failure.",
    )

    # train.py passthrough. Paths default to the GCS FUSE mount.
    parser.add_argument(
        "--config",
        default="configs/sft/default.yaml",
        help="YAML config baked into the image that train.py loads.",
    )
    parser.add_argument("--model-name", default=None, help="Override model_name_or_path.")
    parser.add_argument(
        "--train-path", default=f"/gcs/{BUCKET}/datasets/final-10k/train.jsonl"
    )
    parser.add_argument(
        "--dev-path", default=f"/gcs/{BUCKET}/datasets/final-10k/dev.jsonl"
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


def build_module_args(args: argparse.Namespace, run_name: str) -> list[str]:
    gcs_root = f"/gcs/{args.bucket}"
    module_args: list[str] = []
    if args.module == "src.sft.train":
        module_args += [
            "--config", args.config,
            "--train_path", args.train_path,
            "--dev_path", args.dev_path,
            "--output_dir", f"{gcs_root}/runs",
            "--run_name", run_name,
        ]
        if args.model_name is not None:
            module_args += ["--model_name_or_path", args.model_name]
        if args.max_steps is not None:
            module_args += ["--max_steps", str(args.max_steps)]
    # argparse.REMAINDER keeps a leading "--"; drop it before forwarding.
    extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
    return module_args + extra


def job_resource_name(job: aiplatform.CustomJob) -> str | None:
    with contextlib.suppress(Exception):
        return job.resource_name
    with contextlib.suppress(Exception):
        resource = getattr(job, "_gca_resource", None)
        if resource is not None:
            return getattr(resource, "name", None)
    return None


def main() -> None:
    args = parse_args()
    run_name = args.run_name or default_run_name()
    image_uri = args.image_uri or default_image_uri(args.region)

    aiplatform.init(
        project=PROJECT, location=args.region, staging_bucket=f"gs://{args.bucket}"
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
            "disk_spec": {
                "boot_disk_type": args.boot_disk_type,
                "boot_disk_size_gb": args.boot_disk_gb,
            },
            "container_spec": {
                "image_uri": image_uri,
                "command": ["python", "-m", args.module],
                "args": build_module_args(args, run_name),
                "env": [
                    {"name": "GOOGLE_CLOUD_PROJECT", "value": PROJECT},
                    {"name": "GOOGLE_CLOUD_LOCATION", "value": args.region},
                ],
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=run_name, worker_pool_specs=worker_pool_specs
    )
    print(f"Submitting Vertex CustomJob '{run_name}'")
    print(f"  module: {args.module}")
    print(f"  region: {args.region}")
    print(f"  bucket: gs://{args.bucket}")
    print(f"  image:  {image_uri}")
    if args.module == "src.sft.train":
        print(f"  output: gs://{args.bucket}/runs/ (run dir created by train.py)")
    else:
        print("  args:   forwarded exactly from the extra '-- ...' section")
    if args.no_wait:
        job.submit()
        resource_name = job_resource_name(job)
        print(f"[vertex-submit] Submitted (no-wait): {resource_name}", flush=True)
        print("[vertex-submit] The job runs on Vertex independently of this process.", flush=True)
        return
    try:
        job.run()
    except KeyboardInterrupt:
        resource_name = job_resource_name(job)
        print("[vertex-submit] Interrupted locally (Ctrl+C).", flush=True)
        if resource_name is None:
            print(
                "[vertex-submit] The CustomJob was not submitted yet, so there is nothing to cancel remotely.",
                flush=True,
            )
        else:
            print(f"[vertex-submit] Attempting to cancel remote job: {resource_name}", flush=True)
            with contextlib.suppress(Exception):
                job.cancel()
            print("[vertex-submit] Cancellation requested. Verify the final state in Vertex AI.", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
