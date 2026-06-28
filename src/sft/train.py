from __future__ import annotations

from src.sft.config import config_from_args, parse_args
from src.sft.trainer import train
from src.sft.utils import (
    file_fingerprint,
    git_info,
    make_run_dir,
    package_versions,
    save_command,
    set_reproducible_seed,
    timestamp_run_id,
    write_json,
    write_yaml,
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    set_reproducible_seed(config.seed)
    run_id = timestamp_run_id(config.model_name_or_path, config.loss_type, config.run_name)
    run_dir = make_run_dir(config.output_dir, run_id)

    write_yaml(run_dir / "config.yaml", config.to_dict())
    save_command(run_dir / "command.txt")
    write_json(run_dir / "git_info.json", git_info())
    write_json(
        run_dir / "data_fingerprint.json",
        {
            "train": file_fingerprint(config.train_path),
            "dev": file_fingerprint(config.dev_path),
            "versions": package_versions(),
        },
    )

    state = train(config, run_dir)
    print(f"[sft] finished run_dir={run_dir} global_step={state['global_step']} best_dev_loss={state['best_dev_loss']}")


if __name__ == "__main__":
    main()
