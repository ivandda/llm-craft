import argparse
import json
import os
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_TEMPLATE = (
    "Given two concepts, combine them into one resulting concept.\n\n"
    "Concept A: {input_a}\n"
    "Concept B: {input_b}\n\n"
    "Return only the resulting concept."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local inference with a trained LoRA/QLoRA adapter."
    )
    parser.add_argument(
        "--adapter-dir",
        default="artifacts/sft/local-smoke",
        help="Directory containing the trained adapter files.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional base model override. By default, inferred from adapter_config.json.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Raw user prompt. If omitted, `--input-a` and `--input-b` are used.",
    )
    parser.add_argument(
        "--input-a",
        default=None,
        help="Concept A for the default recipe prompt.",
    )
    parser.add_argument(
        "--input-b",
        default=None,
        help="Concept B for the default recipe prompt.",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_TEMPLATE,
        help="Prompt template used when `--prompt` is not provided.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--lora-mode",
        choices=["lora", "qlora"],
        default="lora",
        help="Use `qlora` only when loading a 4-bit adapter on GPU.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_base_model_name(adapter_dir: str, model_name_override: str | None) -> str:
    if model_name_override:
        return model_name_override

    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(
            f"Could not infer base model because {adapter_config_path} does not exist. "
            "Pass --model-name explicitly."
        )

    with open(adapter_config_path, "r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)

    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            "adapter_config.json does not contain `base_model_name_or_path`. "
            "Pass --model-name explicitly."
        )
    return base_model_name


def build_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.input_a is None or args.input_b is None:
        raise ValueError("Provide either `--prompt` or both `--input-a` and `--input-b`.")
    return args.prompt_template.format(input_a=args.input_a, input_b=args.input_b)


def load_model_and_tokenizer(args: argparse.Namespace):
    base_model_name = resolve_base_model_name(args.adapter_dir, args.model_name)
    dtype = compute_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }

    if args.lora_mode == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA inference requires a CUDA-capable GPU.")
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = dtype

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    if not torch.cuda.is_available():
        model.to("cpu")

    model.eval()
    return model, tokenizer


def generate(args: argparse.Namespace) -> str:
    model, tokenizer = load_model_and_tokenizer(args)
    prompt_text = build_prompt(args)
    messages = [{"role": "user", "content": prompt_text}]

    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if args.temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    prediction = generate(args)
    print(prediction)


if __name__ == "__main__":
    main()
