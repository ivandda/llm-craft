import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_PROMPT_TEMPLATE = (
    "Given two concepts, combine them into one resulting concept.\n\n"
    "Concept A: {input_a}\n"
    "Concept B: {input_b}\n\n"
    "Return only the resulting concept."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible SFT with LoRA or QLoRA on a recipe JSONL dataset."
    )
    parser.add_argument(
        "--model-name",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Base chat model to fine-tune locally or on a GPU VM.",
    )
    parser.add_argument(
        "--train-file",
        default="artifacts/data/recipes_train_sample_8000.jsonl",
        help="Training JSONL file with recipe records or legacy `messages` records.",
    )
    parser.add_argument(
        "--eval-file",
        default=None,
        help="Optional recipe evaluation JSONL file with the same format.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/sft/local-smoke",
        help="Directory where adapters, logs, and trainer state will be written.",
    )
    parser.add_argument(
        "--lora-mode",
        choices=["lora", "qlora"],
        default="lora",
        help="Use standard LoRA or 4-bit QLoRA.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Runtime prompt template for records with input_a/input_b.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Overrides epochs when greater than 0.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
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


def split_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Each row must contain at least one prompt message and one assistant message.")

    assistant_message = messages[-1]
    if assistant_message.get("role") != "assistant":
        raise ValueError("The final message in each row must have role='assistant'.")

    prompt_messages = messages[:-1]
    return prompt_messages, assistant_message


def recipe_outputs(example: dict[str, Any]) -> list[str]:
    if example.get("output"):
        return [example["output"]]
    outputs = example.get("outputs") or example.get("known_outputs")
    if isinstance(outputs, list):
        return [output for output in outputs if output]
    raise ValueError("Recipe rows must contain `output`, `outputs`, or `known_outputs`.")


def render_recipe_prompt(input_a: str, input_b: str, prompt_template: str, tokenizer: Any) -> str:
    prompt_content = prompt_template.format(input_a=input_a, input_b=input_b)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenized_example(prompt_text: str, assistant_text: str, tokenizer: Any, max_length: int) -> dict[str, Any]:
    if tokenizer.eos_token:
        full_text = prompt_text + assistant_text + tokenizer.eos_token
    else:
        full_text = prompt_text + assistant_text

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    labels = list(tokenized["input_ids"])
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


def build_tokenized_dataset(dataset, tokenizer, max_length: int, prompt_template: str = DEFAULT_PROMPT_TEMPLATE):
    def preprocess(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        tokenized_rows: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        batch_size = len(next(iter(batch.values()))) if batch else 0

        for index in range(batch_size):
            example = {key: values[index] for key, values in batch.items()}
            if example.get("messages"):
                prompt_messages, assistant_message = split_messages(example["messages"])
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                targets = [assistant_message["content"]]
            else:
                prompt_text = render_recipe_prompt(
                    input_a=example.get("input_a", ""),
                    input_b=example.get("input_b", ""),
                    prompt_template=prompt_template,
                    tokenizer=tokenizer,
                )
                targets = recipe_outputs(example)

            for target in targets:
                row = tokenized_example(prompt_text, target, tokenizer, max_length)
                tokenized_rows["input_ids"].append(row["input_ids"])
                tokenized_rows["attention_mask"].append(row["attention_mask"])
                tokenized_rows["labels"].append(row["labels"])

        return tokenized_rows

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing recipe SFT data",
    )

    return tokenized.filter(
        lambda example: any(label != -100 for label in example["labels"]),
        desc="Dropping rows where the assistant target was truncated away",
    )


@dataclass
class SFTDataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in features]
        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        padded_labels = []
        sequence_length = batch["input_ids"].shape[1]
        for label in labels:
            padded = label + ([-100] * (sequence_length - len(label)))
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def load_model_and_tokenizer(args: argparse.Namespace):
    dtype = compute_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }

    if args.lora_mode == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a CUDA-capable GPU.")
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.lora_mode == "qlora":
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        bias="none",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer, dtype


def build_training_args(args: argparse.Namespace, dtype: torch.dtype, has_eval: bool) -> TrainingArguments:
    os.makedirs(args.output_dir, exist_ok=True)

    return TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if has_eval else None,
        eval_strategy="steps" if has_eval else "no",
        save_strategy="steps",
        bf16=torch.cuda.is_available() and dtype == torch.bfloat16,
        fp16=torch.cuda.is_available() and dtype == torch.float16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if args.lora_mode == "qlora" else "adamw_torch",
    )


def save_run_config(args: argparse.Namespace) -> None:
    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    if args.eval_file and not os.path.exists(args.eval_file):
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")

    model, tokenizer, dtype = load_model_and_tokenizer(args)
    model.print_trainable_parameters()

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["eval"] = args.eval_file

    raw_datasets = load_dataset("json", data_files=data_files)
    train_dataset = build_tokenized_dataset(raw_datasets["train"], tokenizer, args.max_length, args.prompt_template)
    eval_dataset = None
    if args.eval_file:
        eval_dataset = build_tokenized_dataset(raw_datasets["eval"], tokenizer, args.max_length, args.prompt_template)

    training_args = build_training_args(args, dtype, eval_dataset is not None)
    save_run_config(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
