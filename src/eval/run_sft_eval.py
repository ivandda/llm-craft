import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from src.eval.metrics import evaluate_prediction


DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_TEMPLATE = (
    "Given two concepts, combine them into one resulting concept.\n\n"
    "Concept A: {input_a}\n"
    "Concept B: {input_b}\n\n"
    "Return only the resulting concept."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch evaluation for a base model or a trained LoRA/QLoRA adapter."
    )
    parser.add_argument(
        "--eval-file",
        default="datasets/processed/eval_dev_1k.jsonl",
        help="JSONL file with input_a, input_b, canonical_output, and known_outputs.",
    )
    parser.add_argument(
        "--output-file",
        default="artifacts/eval/sft_eval_predictions.jsonl",
        help="Where prediction records will be written as JSONL.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional trained PEFT adapter directory. If omitted, evaluates the base model.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Base model name. With --adapter-dir, this overrides adapter_config.json.",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_TEMPLATE,
        help="Prompt template used to render each eval pair.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples to evaluate for smoke tests.",
    )
    parser.add_argument(
        "--lora-mode",
        choices=["lora", "qlora"],
        default="lora",
        help="Use qlora only when loading a 4-bit adapter on GPU.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def build_prompt(input_a: str, input_b: str, prompt_template: str) -> str:
    return prompt_template.format(input_a=input_a, input_b=input_b)


def iter_eval_records(eval_file: str, limit: int | None = None) -> Iterable[dict[str, Any]]:
    seen = 0
    with open(eval_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            seen += 1
            if limit is not None and seen >= limit:
                return


def build_output_record(eval_record: dict[str, Any], prediction: str) -> dict[str, Any]:
    known_outputs = eval_record.get("known_outputs", [])
    evaluation = evaluate_prediction(
        prediction=prediction,
        canonical_output=eval_record.get("canonical_output", ""),
        known_outputs=known_outputs,
    )

    return {
        "pair_id": eval_record.get("pair_id"),
        "input_a": eval_record.get("input_a"),
        "input_b": eval_record.get("input_b"),
        "prediction": prediction,
        "canonical_output": eval_record.get("canonical_output"),
        "known_outputs": known_outputs,
        "exact_canonical_match": evaluation.exact_canonical_match,
        "known_output_match": evaluation.known_output_match,
        "is_empty_prediction": evaluation.is_empty_prediction,
    }


def summarize_output_records(records: list[dict[str, Any]]) -> dict[str, int | float]:
    num_examples = len(records)
    if num_examples == 0:
        return {
            "num_examples": 0,
            "canonical_accuracy": 0.0,
            "known_output_accuracy": 0.0,
            "empty_predictions": 0,
        }

    canonical_matches = sum(1 for record in records if record["exact_canonical_match"])
    known_matches = sum(1 for record in records if record["known_output_match"])
    empty_predictions = sum(1 for record in records if record["is_empty_prediction"])

    return {
        "num_examples": num_examples,
        "canonical_accuracy": canonical_matches / num_examples,
        "known_output_accuracy": known_matches / num_examples,
        "empty_predictions": empty_predictions,
    }


def _compute_dtype(torch_module: Any) -> Any:
    if not torch_module.cuda.is_available():
        return torch_module.float32
    if torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16
    return torch_module.float16


def _resolve_base_model_name(adapter_dir: str, model_name: str | None) -> str:
    if model_name:
        return model_name

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


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = _compute_dtype(torch)
    base_model_name = _resolve_base_model_name(args.adapter_dir, args.model_name) if args.adapter_dir else args.model_name
    tokenizer_source = args.adapter_dir or base_model_name

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    uses_device_map = False

    if args.lora_mode == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA evaluation requires a CUDA-capable GPU.")
        uses_device_map = True
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    if torch.cuda.is_available() and not uses_device_map:
        model.to("cuda")
    elif not torch.cuda.is_available():
        model.to("cpu")

    model.eval()
    return model, tokenizer


def generate_prediction(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    messages = [{"role": "user", "content": prompt_text}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def run_evaluation(args: argparse.Namespace) -> dict[str, int | float | str]:
    if not os.path.exists(args.eval_file):
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")

    model, tokenizer = load_model_and_tokenizer(args)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for eval_record in iter_eval_records(args.eval_file, args.limit):
            prompt_text = build_prompt(
                input_a=eval_record.get("input_a", ""),
                input_b=eval_record.get("input_b", ""),
                prompt_template=args.prompt_template,
            )
            prediction = generate_prediction(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            output_record = build_output_record(eval_record, prediction)
            output_records.append(output_record)
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    summary = summarize_output_records(output_records)
    return {
        **summary,
        "output_file": args.output_file,
    }


def main() -> None:
    args = parse_args()
    summary = run_evaluation(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
