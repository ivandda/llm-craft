from datasets import load_dataset
ds = load_dataset("ericlewis/infinite-craft-recipes", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl", "test": "data/test.jsonl"})
