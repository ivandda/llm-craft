---
license: mit
task_categories:
  - text-generation
language:
  - en
size_categories:
  - 10K<n<100K
tags:
  - infinite-craft
  - element-fusion
  - game-ai
---

# Infinite Craft Recipes Dataset

Training data for fine-tuning element fusion models, based on the Infinite Craft game by neal.fun.

## Dataset Structure

Each example is a chat-format message with:
- **System**: "Fuse two elements into one. Output JSON only."
- **User**: "Element1 + Element2"  
- **Assistant**: `{"result": "Name", "emoji": "symbol"}`

## Splits

| Split | Examples | Description |
|-------|----------|-------------|
| train | 20,000 | Training set |
| val | 2,000 | Validation set |
| test | 2,000 | Held-out test set |

## Source

Derived from 205,626 decoded recipes from community recipe dumps (vantezzen collection). Randomly sampled and split with no overlap.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("ericlewis/infinite-craft-recipes", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl", "test": "data/test.jsonl"})
```

## Format

```json
{
  "messages": [
    {"role": "system", "content": "Fuse two elements into one. Output JSON only.\n{\"result\": \"name\", \"emoji\": \"symbol\"}"},
    {"role": "user", "content": "Water + Fire"},
    {"role": "assistant", "content": "{\"result\": \"Steam\", \"emoji\": \"💨\"}"}
  ]
}
```
