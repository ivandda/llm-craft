import os
import sys
import time

# Add the project root directory to the python path to allow importing src modules
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.data.normalize import main as normalize_main
from src.data.clean import main as clean_main
from src.data.export_sft import main as export_sft_main
from src.data.export_eval import main as export_eval_main

def print_separator(title: str):
    print("=" * 60)
    print(f" >>> STEP: {title.upper()} <<<")
    print("=" * 60)

def main():
    print("============================================================")
    print("         INICIANDO PIPELINE DE DATOS: LLM-CRAFT             ")
    print("============================================================")
    start_time = time.time()

    # Step 1: Normalization
    print_separator("1. Normalización de Datos Raw")
    step_start = time.time()
    normalize_main()
    print(f"Paso 1 completado en {time.time() - step_start:.2f} segundos.\n")

    # Step 2: Cleaning and Splits
    print_separator("2. Limpieza, Agregación y Particionado (Splits)")
    step_start = time.time()
    clean_main()
    print(f"Paso 2 completado en {time.time() - step_start:.2f} segundos.\n")

    # Step 3: SFT Export
    print_separator("3. Exportación de Dataset SFT")
    step_start = time.time()
    export_sft_main()
    print(f"Paso 3 completado en {time.time() - step_start:.2f} segundos.\n")

    # Step 4: Evaluation Export
    print_separator("4. Exportación de Conjuntos de Evaluación")
    step_start = time.time()
    export_eval_main()
    print(f"Paso 4 completado en {time.time() - step_start:.2f} segundos.\n")

    total_time = time.time() - start_time
    print("============================================================")
    print(f"  PIPELINE COMPLETADO CON ÉXITO EN {total_time:.2f} SEGUNDOS")
    print("============================================================")

if __name__ == "__main__":
    main()
