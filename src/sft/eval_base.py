from __future__ import annotations

from src.sft.config import config_from_args, parse_args
from src.sft.trainer import build_dataloaders, build_model_and_tokenizer, evaluate
from src.sft.utils import set_reproducible_seed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    set_reproducible_seed(config.seed)

    from accelerate import Accelerator

    accelerator = Accelerator(
        mixed_precision="bf16" if config.bf16 else "fp16" if config.fp16 else "no",
    )
    model, tokenizer = build_model_and_tokenizer(config, apply_lora=False)
    _, dev_loader = build_dataloaders(config, tokenizer)
    model, dev_loader = accelerator.prepare(model, dev_loader)

    dev_loss = evaluate(model, dev_loader, config, accelerator)
    if accelerator.is_main_process:
        print(
            "[sft] base_model_validation "
            f"model={config.model_name_or_path} "
            f"dev_recipes={len(dev_loader.dataset)} "
            f"dev_loss={dev_loss:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
