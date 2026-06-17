import os
from datasets import load_dataset

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to local files
train_path = os.path.join(current_dir, "data", "train.jsonl")
val_path = os.path.join(current_dir, "data", "val.jsonl")
test_path = os.path.join(current_dir, "data", "test.jsonl")

# Load dataset locally
ds = load_dataset(
    "json",
    data_files={
        "train": train_path,
        "validation": val_path,
        "test": test_path
    }
)

if __name__ == "__main__":
    print(ds)
    print("Example training item:", ds["train"][0])
