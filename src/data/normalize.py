import os
import json
import glob
import csv
from typing import Dict, Optional, List, Tuple
from src.data.schemas import RecipeObservation

# Globals
emoji_map: Dict[str, str] = {}

def register_emoji(name: str, emoji: Optional[str]):
    if not name or not emoji:
        return
    name_clean = name.strip().lower()
    if name_clean not in emoji_map:
        emoji_map[name_clean] = emoji
    elif emoji_map[name_clean] == "⚪" and emoji != "⚪":
        emoji_map[name_clean] = emoji

def get_emoji(name: str) -> Optional[str]:
    return emoji_map.get(name.strip().lower())

def parse_ericlewis(base_dir: str) -> List[Tuple[str, str, str, Optional[str], str]]:
    recipes = []
    data_dir = os.path.join(base_dir, "datasets", "eirclewis", "data")
    for filename in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found.")
            continue
        
        source = f"ericlewis_{filename.split('.')[0]}"
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    messages = {msg["role"]: msg["content"] for msg in obj.get("messages", [])}
                    user_content = messages.get("user", "")
                    assistant_content = messages.get("assistant", "")
                    
                    parts = user_content.split(" + ")
                    if len(parts) != 2:
                        continue
                    input_a, input_b = parts[0].strip(), parts[1].strip()
                    
                    try:
                        res_obj = json.loads(assistant_content)
                        output = res_obj.get("result", "").strip()
                        emoji_output = res_obj.get("emoji", "").strip()
                    except Exception:
                        output = assistant_content.strip()
                        emoji_output = None
                    
                    if input_a and input_b and output:
                        recipes.append((input_a, input_b, output, emoji_output, source))
                        register_emoji(output, emoji_output)
                except Exception as e:
                    print(f"Error parsing ericlewis line: {e}")
    return recipes

def parse_elementia(base_dir: str) -> List[Tuple[str, str, str, Optional[str], str]]:
    recipes = []
    file_path = os.path.join(base_dir, "datasets", "elementia", "recipes.csv")
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return recipes

    source = "elementia"
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                continue
            combination, result = row[0], row[1]
            parts = combination.split("+")
            if len(parts) != 2:
                continue
            input_a, input_b = parts[0].strip(), parts[1].strip()
            output = result.strip()
            if input_a and input_b and output:
                recipes.append((input_a, input_b, output, None, source))
    return recipes

def parse_expitau(base_dir: str) -> List[Tuple[str, str, str, Optional[str], str]]:
    recipes = []
    file_path = os.path.join(base_dir, "datasets", "expitau", "web/data/data.json")
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return recipes

    source = "expitau"
    with open(file_path, "r", encoding="utf-8") as f:
        data_obj = json.load(f)
        
    index = data_obj.get("index", {})
    id_map = {}
    for item_id, details in index.items():
        if len(details) >= 2:
            emoji, name = details[0], details[1]
            id_map[item_id] = (name.strip(), emoji.strip() if emoji else None)
            register_emoji(name, emoji)
            
    raw_data = data_obj.get("data", "")
    raw_recipes = raw_data.split(";")
    for raw_rec in raw_recipes:
        if not raw_rec:
            continue
        parts = raw_rec.split(",")
        if len(parts) != 3:
            continue
        id_a, id_b, id_out = parts[0], parts[1], parts[2]
        if id_a in id_map and id_b in id_map and id_out in id_map:
            name_a, emoji_a = id_map[id_a]
            name_b, emoji_b = id_map[id_b]
            name_out, emoji_out = id_map[id_out]
            if name_a and name_b and name_out:
                recipes.append((name_a, name_b, name_out, emoji_out, source))
    return recipes

def parse_redfast00(base_dir: str) -> List[Tuple[str, str, str, Optional[str], str]]:
    recipes = []
    dir_path = os.path.join(base_dir, "datasets", "redfast00", "JSONrecipes")
    if not os.path.exists(dir_path):
        print(f"Warning: Directory {dir_path} not found.")
        return recipes

    json_files = glob.glob(os.path.join(dir_path, "*.json"))
    for file_path in json_files:
        source = f"redfast00_{os.path.basename(file_path).replace('.json', '')}"
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            
        names = obj.get("names", {})
        id_to_name = {}
        for k, v in names.items():
            try:
                id_to_name[int(k)] = v.strip()
            except ValueError:
                id_to_name[k] = v.strip()
                
        raw_recipes = obj.get("recipes", [])
        for rec in raw_recipes:
            ingredients = rec.get("ingredients", [])
            results = rec.get("results", [])
            if len(ingredients) != 2:
                continue
            id_a, id_b = ingredients[0], ingredients[1]
            name_a = id_to_name.get(id_a) or id_to_name.get(str(id_a))
            name_b = id_to_name.get(id_b) or id_to_name.get(str(id_b))
            
            if not name_a or not name_b:
                continue
                
            for res_id in results:
                name_out = id_to_name.get(res_id) or id_to_name.get(str(res_id))
                if name_a and name_b and name_out:
                    recipes.append((name_a, name_b, name_out, None, source))
    return recipes

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    print(f"Project base directory: {base_dir}")

    # Step 1: Parse all datasets to collect raw recipes and register emojis
    print("Parsing ericlewis dataset...")
    ericlewis_raw = parse_ericlewis(base_dir)
    print(f"Parsed {len(ericlewis_raw)} recipes from ericlewis.")

    print("Parsing elementia dataset...")
    elementia_raw = parse_elementia(base_dir)
    print(f"Parsed {len(elementia_raw)} recipes from elementia.")

    print("Parsing expitau dataset...")
    expitau_raw = parse_expitau(base_dir)
    print(f"Parsed {len(expitau_raw)} recipes from expitau.")

    print("Parsing redfast00 dataset...")
    redfast00_raw = parse_redfast00(base_dir)
    print(f"Parsed {len(redfast00_raw)} recipes from redfast00.")

    # Combine all raw recipes
    all_raw = ericlewis_raw + elementia_raw + expitau_raw + redfast00_raw
    print(f"Total raw recipe observations parsed: {len(all_raw)}")

    # Step 2: Canonicalize and enrich with emojis
    processed_observations: List[RecipeObservation] = []
    for input_a, input_b, output, emoji_out, source in all_raw:
        # Sort ingredients alphabetically for commutative consistency
        if input_a.strip().lower() <= input_b.strip().lower():
            sorted_a, sorted_b = input_a.strip(), input_b.strip()
        else:
            sorted_a, sorted_b = input_b.strip(), input_a.strip()

        # Lookup/backfill emojis from the emoji_map
        emoji_a = get_emoji(sorted_a)
        emoji_b = get_emoji(sorted_b)
        # Use provided output emoji or look it up
        final_emoji_out = emoji_out or get_emoji(output)

        obs = RecipeObservation(
            input_a=sorted_a,
            input_b=sorted_b,
            output=output.strip(),
            emoji_a=emoji_a,
            emoji_b=emoji_b,
            emoji_output=final_emoji_out,
            source=source
        )
        processed_observations.append(obs)

    # Step 3: Write out to datasets/processed/recipe_observations_v0.jsonl
    processed_dir = os.path.join(base_dir, "datasets", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out_file = os.path.join(processed_dir, "recipe_observations_v0.jsonl")
    
    print(f"Writing observations to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for obs in processed_observations:
            f.write(json.dumps(obs.to_dict(), ensure_ascii=False) + "\n")
            
    print("Normalization complete.")

if __name__ == "__main__":
    main()
